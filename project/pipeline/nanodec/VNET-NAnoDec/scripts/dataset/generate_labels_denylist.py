#! /usr/bin/python

#configuration
FLOW_DIR_CONF = "/data/kinit/flows/"
GROUND_TRUTH_FILE_CONF = "/data/kinit/attack_records_parseable"
DENYLIST_DIR_CONF = '/data/kinit/Deny_list/'

#import modules
import sys
import os
import os.path
import pandas as pd
from datetime import datetime
import numpy as np
#import pathlib
#import time

#
# Find labels for attack flow records in dirpath+filename according to the ground truth gt
# Saves the labeled flow records in a new directory modified from flowdir with all files to be labeled
#
def find_label(gt, dirpath, filename, flowdir, deny_list):
  FEATURES_CASTDICT = {'IN_BYTES': 'uint32', 'IN_PKTS': 'uint32', 'PROTOCOL': 'uint8', 'TCP_FLAGS': 'uint8', 'L4_SRC_PORT': 'uint16', 'IPV4_SRC_ADDR': 'string', 'IPV6_SRC_ADDR': 'string', 'L4_DST_PORT': 'uint16', 'IPV4_DST_ADDR': 'string', 'IPV6_DST_ADDR': 'string', 'OUT_BYTES': 'uint32', 'OUT_PKTS': 'uint32', 'MIN_IP_PKT_LEN': 'uint16', 'MAX_IP_PKT_LEN': 'uint16', 'ICMP_TYPE': 'uint16', 'MIN_TTL': 'uint8', 'MAX_TTL': 'uint8', 'DIRECTION': 'uint8', 'FLOW_START_MILLISECONDS': 'uint64', 'FLOW_END_MILLISECONDS': 'uint64', 'SRC_FRAGMENTS': 'uint16', 'DST_FRAGMENTS': 'uint16', 'CLIENT_TCP_FLAGS': 'uint8', 'SERVER_TCP_FLAGS': 'uint8', 'SRC_TO_DST_AVG_THROUGHPUT': 'uint32', 'DST_TO_SRC_AVG_THROUGHPUT': 'uint32', 'NUM_PKTS_UP_TO_128_BYTES': 'uint32', 'NUM_PKTS_128_TO_256_BYTES': 'uint32', 'NUM_PKTS_256_TO_512_BYTES': 'uint32', 'NUM_PKTS_512_TO_1024_BYTES': 'uint32', 'NUM_PKTS_1024_TO_1514_BYTES': 'uint32', 'NUM_PKTS_OVER_1514_BYTES': 'uint32', 'LONGEST_FLOW_PKT': 'uint32', 'SHORTEST_FLOW_PKT': 'uint32', 'RETRANSMITTED_IN_PKTS': 'uint32', 'RETRANSMITTED_OUT_PKTS': 'uint32', 'OOORDER_IN_PKTS': 'uint32', 'OOORDER_OUT_PKTS': 'uint32', 'DURATION_IN': 'uint32', 'DURATION_OUT': 'uint32', 'TCP_WIN_MIN_IN': 'uint16', 'TCP_WIN_MAX_IN': 'uint16', 'TCP_WIN_MSS_IN': 'uint16', 'TCP_WIN_SCALE_IN': 'uint8', 'TCP_WIN_MIN_OUT': 'uint16', 'TCP_WIN_MAX_OUT': 'uint16', 'TCP_WIN_MSS_OUT': 'uint16', 'TCP_WIN_SCALE_OUT': 'uint8', 'FLOW_VERDICT': 'uint16', 'SRC_TO_DST_IAT_MIN': 'uint16', 'SRC_TO_DST_IAT_MAX': 'uint16', 'SRC_TO_DST_IAT_AVG': 'uint16', 'SRC_TO_DST_IAT_STDDEV': 'uint16', 'DST_TO_SRC_IAT_MIN': 'uint16', 'DST_TO_SRC_IAT_MAX': 'uint16', 'DST_TO_SRC_IAT_AVG': 'uint16', 'DST_TO_SRC_IAT_STDDEV': 'uint16', 'APPLICATION_ID': 'int32'}
  
  try:
    df = pd.read_csv(os.path.join(dirpath, filename), delimiter='|', compression='gzip', dtype=FEATURES_CASTDICT)
  except Exception:
    print('Cannot decompress: '+ os.path.join(dirpath, filename))
    return
  df.drop_duplicates(inplace=True)
  
  #find all attacks in file
  #either source or destination IP matches ground truth
  df_sip = df[df['IPV4_SRC_ADDR'] == '46.229.235.84'].merge(gt[['Label', 'IPV4_SRC_ADDR', 'ts', 'te']], on=['IPV4_SRC_ADDR'], how='left')
  df_dip = df[df['IPV4_DST_ADDR'] == '46.229.235.84'].merge(gt[['Label', 'IPV4_DST_ADDR', 'ts', 'te']], on=['IPV4_DST_ADDR'], how='left')
  attacks = pd.concat([df_sip, df_dip])
  attacks = attacks[(attacks['FLOW_START_MILLISECONDS']/1000 <= attacks.te) & (attacks['FLOW_END_MILLISECONDS']/1000 >= attacks.ts)]
  attacks.drop(columns=['ts', 'te'], inplace=True)
  
  #label original dataframe
  df2 = df.merge(attacks, on=df.columns.to_list(), how='left', copy=False) #identified attacks according the ground truth
  df2.loc[df2['Label'].isnull(), 'Label'] = "background" #not attacks
  df2.loc[((df2['IPV4_DST_ADDR'] == '86.110.242.235') | (df2['IPV4_SRC_ADDR'] == '86.110.242.235')), 'Label'] = "C&C_1" #command and control communication of C&C server with attackers (not present in the ground truth)

  #add denylist info
  df2 = detect_deny(df2, deny_list, 4)

  #add aditional features
  df2['Background_filter'] = pd.NA
  
  #save the result
  newflowdir = flowdir
  if ('/flows/' in newflowdir):
    newflowdir = newflowdir.replace('/flows/', '/flows_labeled/')
  else:
    newflowdir = flowdir[:-1] + '_labeled/'
  filepath = os.path.join(dirpath, filename).replace(flowdir, newflowdir)
  os.makedirs(filepath[:filepath.rindex(os.path.sep)], exist_ok=True)
  #time.sleep(1) #wait for OS to create directory
  #pathlib.Path(dirpath.replace(flowdir, newflowdir)).mkdir(parents=True, exist_ok=True)

  #~34.2s per file
  #df2.to_csv(filepath.replace('.flows.gz','.flows.gz'), index=False, compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})
  #~15.8s per file
  #df2.to_feather(filepath.replace('.flows.gz','.flows.feather'), compression='zstd')
  #~28s per file
  df2.to_pickle(filepath.replace('.flows.gz','.flows.pkl-zip'), compression='zip')

#
# Load and prepare ground truth in a file gt_file
# Retrurn: ground truth dataframe gt
#
def prepare_gt(gt_file):

  columns = ['Label', 'tool', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'dp', 'ts', 'te']
  gt = pd.read_csv(gt_file, delimiter='\\', header=None, names=columns)
  gt.drop(columns=['tool', 'dp'], inplace=True)
  gt['IPV4_SRC_ADDR'] = "46.229.235.84" #assign victims address even to the source due to flow records might be reversed
  gt['IPV4_DST_ADDR'] = "46.229.235.84"
  
  #attack label cleaning
  to_be_removed = '|'.join(['Friday', 'Saturday', 'Tuesday', 'Wednesday', 'Thursday', ' morning ', ' afternoon ', ' night ', 'advanced ', ' - minimal aggresivity', ' - low aggresivity', ' - medium aggresivity', ' - high aggresivity'])
  gt['Label'] = gt['Label'].str.replace(to_be_removed,'', regex=True)
  gt['Label'] = gt['Label'].str.strip()
  gt['Label'] = gt['Label']+'_1'

  return gt

#Creates columns fod dataframe for IP deny lits
def ipsum (path,dire,filename):
    format_data = "%Y %m %d"
    a=dire.find('_')
    str1= dire[a+1:a+5] + ' ' + dire[a+5:a+7] + ' ' + dire[a+7:a+9]   
    date=datetime.strptime (str1,format_data)
    filepath=os.path.join(os.path.join(path,dire),filename)
    #print (filepath)
    df=pd.read_csv(filepath, names =['IP', 'Level'], header=None)
    df['Date']=date
    df['Month']=int(dire[a+5:a+7])
    df['Day']=int(dire[a+7:a+9])
    #print (str1,date)
    return df

#load files that contains deny lists
def load_deny_list (name, folder):
    df=pd.DataFrame()
    print ('Loading deny files:')
    for (root, dirs, file) in os.walk(folder):
        for d in dirs:
            if 'ipsum' in d:
                print(root,d)      
                df_help=ipsum (root,d,name)
                df = pd.concat([df,df_help],ignore_index=True)
    print ('End loading')
    denylist_dtypes = {'IP': 'string', 'Level': 'uint8', 'Month': 'uint8', 'Day': 'uint8'}
    return df.astype(denylist_dtypes)

#detect if any IP in df is also on deny list (deny_list). IP_address on deny_list has to be noticed at least "level" times
def detect_deny (df, deny_list, level):
    df['SRC_DENY']=np.NaN
    df['DST_DENY']=np.NaN
    df['Day']=df['FLOW_START_MILLISECONDS'].apply(lambda x: datetime.fromtimestamp(x/1000.0).day)
    for i in df['Day'].unique():
        df.loc[(df['Day']==i) & (df['IPV6_SRC_ADDR']!='::'), ['SRC_DENY']]=df[(df['Day']==i)& (df['IPV6_SRC_ADDR']!='::')]['IPV6_SRC_ADDR'].isin(deny_list.loc[(deny_list['Day']==i) & (deny_list['Level']>4)]['IP'])
        df.loc[(df['Day']==i) & (df['IPV4_SRC_ADDR']!='0.0.0.0'), ['SRC_DENY']]=df[(df['Day']==i)& (df['IPV4_SRC_ADDR']!='0.0.0.0')]['IPV4_SRC_ADDR'].isin(deny_list.loc[(deny_list['Day']==i) & (deny_list['Level']>4)]['IP'])
        df.loc[(df['Day']==i) & (df['IPV6_DST_ADDR']!='::'), ['DST_DENY']]=df[(df['Day']==i)& (df['IPV6_DST_ADDR']!='::')]['IPV6_DST_ADDR'].isin(deny_list.loc[(deny_list['Day']==i) & (deny_list['Level']>4)]['IP'])
        df.loc[(df['Day']==i) & (df['IPV4_DST_ADDR']!='0.0.0.0'), ['DST_DENY']]=df[(df['Day']==i)& (df['IPV4_DST_ADDR']!='0.0.0.0')]['IPV4_DST_ADDR'].isin(deny_list.loc[(deny_list['Day']==i) & (deny_list['Level']>4)]['IP'])
    df=df.drop('Day',axis=1)
    return df
			
# Main function
def main():
  FLOW_DIR = FLOW_DIR_CONF
  GROUND_TRUTH_FILE = GROUND_TRUTH_FILE_CONF
  
  #check whether to use arguments or global configuration variables for directory with flows and groud truth file
  if len(sys.argv) != 3:
    print ("Usage: %s flows_folder/ attack_records_parseable" % sys.argv[0])
    print ("Using preconfigured values")
  else:
    FLOW_DIR = sys.argv[1]
    GROUND_TRUTH_FILE = sys.argv[2]
  
  #preparing ground truth dataframe
  gt = prepare_gt(GROUND_TRUTH_FILE)

  #load deny lists
  deny_list = load_deny_list('raw_ips.csv', DENYLIST_DIR_CONF)

  #for each file in all subdirectories of a given directory FLOW_DIR generate labeled file in a directory FLOW_DIR appended by '_labeled'
  for dirpath, dirnames, filenames in os.walk(FLOW_DIR):
    for filename in [f for f in filenames if f.endswith(".flows.gz")]:
      find_label(gt, dirpath, filename, FLOW_DIR, deny_list)
    
# program start in main
if __name__ == '__main__':
    main()