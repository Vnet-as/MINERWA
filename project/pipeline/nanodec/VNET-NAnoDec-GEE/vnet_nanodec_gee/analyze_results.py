import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

import click


def plot_roc(df: pd.DataFrame, malicious_type: str):
    df['blabel'] = -1
    if malicious_type != "all":
        part_df = df[(df['label'].str.contains('background')) | (df['label'].str.contains(malicious_type))]
    else:
        part_df = df

    part_df.loc[part_df.label.str.contains('background'),'blabel'] = 0
    part_df.loc[part_df.blabel != 0,'blabel'] = 1
    label = (part_df.blabel.tolist())

    mse_loss = part_df.mse_loss.tolist()
    bce_loss = part_df.bce_loss.tolist()
    bce_kd_loss = part_df['bce+kd_loss'].tolist()

    fig, ax = plt.subplots(figsize=(5, 5))

    fpr_mse, tpr_mse, thresholds_mse = metrics.roc_curve(label, mse_loss)
    fpr_bce, tpr_bce, thresholds_bce = metrics.roc_curve(label, bce_loss)
    fpr_bce_kd, tpr_bce_kd, thresholds_bce_kd = metrics.roc_curve(label, bce_kd_loss)

    auc_mse = metrics.auc(fpr_mse, tpr_mse)
    auc_bce = metrics.auc(fpr_bce, tpr_bce)
    auc_bce_kd = metrics.auc(fpr_bce_kd, tpr_bce_kd)

    ax.plot([0, 1], [0,1], 'k--')
    ax.plot(fpr_mse, tpr_mse, label=f'with mse loss (auc = {auc_mse: .4f})')
    #ax.plot(fpr_bce, tpr_bce, label=f'with bce loss (auc = {auc_bce: .4f})')
    #ax.plot(fpr_bce_kd, tpr_bce_kd, label=f'with bce+kd loss (auc = {auc_bce_kd: .4f})')

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    ax.set_title(f'ROC of background and {malicious_type}')

    ax.legend(loc='lower right')

    return fig


def plot_kde(df: pd.DataFrame, loss_type: str, malicious_type: str):
    loss_type_set = {'mse', 'bce', 'bce+kd'}
    if loss_type not in loss_type_set:
        raise Exception(f'Invalid loss_type, only supporting the following: "{loss_type}"')

    normal_recon_error = df[df['label'].str.contains('background')][f'{loss_type}_loss'].tolist()
    if malicious_type == "all":
        malicious_recon_error = df[~df['label'].str.contains('background')][f'{loss_type}_loss'].tolist()
    else:
        malicious_recon_error = df[df['label'].str.contains(malicious_type)][f'{loss_type}_loss'].tolist()
        
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(
        normal_recon_error,
        ax=ax,
        label=f'background {loss_type} loss'
    )
    sns.kdeplot(
        malicious_recon_error,
        ax=ax,
        label=f'{malicious_type} {loss_type} loss'
    )

    ax.set_title(f'Reconstruction Error Distribution of background traffic and {malicious_type}')
    ax.legend(loc='lower right')

    return fig


def plot_hist(df: pd.DataFrame, loss_type: str, malicious_type: str):
    loss_type_set = {'mse', 'bce', 'bce+kd'}
    if loss_type not in loss_type_set:
        raise Exception(f'Invalid loss_type, only supporting the following: "{loss_type}"')

    normal_recon_error = df[df['label'].str.contains('background')][f'{loss_type}_loss'].tolist()
    if malicious_type == "all":
        malicious_recon_error = df[~df['label'].str.contains('background')][f'{loss_type}_loss'].tolist()
    else:
        malicious_recon_error = df[df['label'].str.contains(malicious_type)][f'{loss_type}_loss'].tolist()
    
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.hist([normal_recon_error, malicious_recon_error], bins=100, range=(0, 1.0), label=['Background', malicious_type])
    ax.set_ylim(0, 100)
    ax.set_xlabel(loss_type)
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')

    return fig


@click.command()
@click.option('-r', '--results_filepath', help='path to file containing results; must also contain labels', required=True)
@click.option('-t', '--thresholds_filepath',
              help='path to a CSV file in which thresholds are located or will be generated according to --analysis_mode',
              required=True)
@click.option('-m', '--analysis_mode',
              help=('result analysis mode;'
                    ' "validation" triggers threshold tuning and saves them in --thresholds_filepath;'
                    ' "test" uses existing thresholds from --thresholds_filepath'),
              type=click.Choice(['validation', 'test']),
              default='validation',
              required=False)
def main(
        results_filepath: str,
        thresholds_filepath: str,
        analysis_mode: str,
):
    # initialise logger
    logger = logging.getLogger(__file__)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    df = pd.read_parquet(results_filepath)
    
    df.mse_loss = df.mse_loss.astype('float32')
    df.bce_loss = df.bce_loss.astype('float32')
    df['bce+kd_loss'] = df['bce+kd_loss'].astype('float32')
    df.label = df.label.astype('string')

    if 'filter_pred' not in df.columns.to_list():
        logger.info("Classification-based filter results not included in the results file. Exiting...")
        sys.exit(1)
        
    df.filter_pred = df.filter_pred.astype('uint8')
    df.filter_prob = df.filter_prob.astype('float16')

    # threshold tuning
    if analysis_mode == 'validation':
        val_mse_mean_0 = df[df.label.str.contains('background')].mse_loss.mean()
        val_mse_std_0 = df[df.label.str.contains('background')].mse_loss.std()
        logger.info(val_mse_mean_0, val_mse_std_0)
        
        filter_thresholds = [0.8, 0.85, 0.9, 0.95, 1.0]
        detector_thresholds = [
            val_mse_mean_0 + val_mse_std_0/3,
            val_mse_mean_0 + val_mse_std_0/2,
            val_mse_mean_0 + val_mse_std_0,
            val_mse_mean_0 + val_mse_std_0*2,
            val_mse_mean_0 + val_mse_std_0*3,
        ]
        
        df_orig = df
        best_auc = 0
        best_ft = None
        best_dt = None
        
        for reconstruction_error_threshold in detector_thresholds:
          for known_attack_confidence_threshold in filter_thresholds:
            df = df_orig[['mse_loss', 'label', 'filter_pred', 'filter_prob']].copy()
            df.loc[((df.filter_pred == 1) & (df.filter_prob >= known_attack_confidence_threshold)), 'mse_loss'] = 1.0
            binary_labels = df['label'].copy().astype('object')
            binary_labels.loc[binary_labels == 'background'] = int(0)
            binary_labels.loc[binary_labels != 0] = int(1)
            binary_predictions = df['filter_pred'].copy()
            binary_predictions.loc[:] = int(0)
            binary_predictions.loc[df.mse_loss >= reconstruction_error_threshold] = int(1)
            auc = metrics.roc_auc_score(list(binary_labels), list(binary_predictions))
            if auc > best_auc:
              best_auc = auc
              best_ft = known_attack_confidence_threshold
              best_dt = reconstruction_error_threshold
            logger.info('Filter threshold: ', known_attack_confidence_threshold)
            logger.info('Detector threshold: ', reconstruction_error_threshold)
            logger.info('AUC:', metrics.roc_auc_score(list(binary_labels), list(binary_predictions)))
            logger.info(metrics.classification_report(list(binary_labels), list(binary_predictions), digits=6))
            logger.info(metrics.confusion_matrix(list(binary_labels), list(binary_predictions)))
            logger.info('----------------------------------------------------')
        df = df_orig
        thresholds = pd.DataFrame({'filter_threshold': best_ft, 'detector_threshold': best_dt}, index=[0])
        thresholds.to_csv(thresholds_filepath, index=False)

    # load thresholds and analyze results with filter
    thresholds = pd.read_csv(thresholds_filepath)
    known_attack_confidence_threshold = thresholds['filter_threshold'][0]
    reconstruction_error_threshold = thresholds['detector_threshold'][0]

    logger.info('Filter confidence threshold = ', known_attack_confidence_threshold)
    df.loc[((df.filter_pred == 1) & (df.filter_prob >= known_attack_confidence_threshold)), 'mse_loss'] = 1.0

    results_filepath = results_filepath.replace(results_filepath.split('/')[-1], '')
    plot_roc(df, "all").savefig(results_filepath + 'filter_all_auc-roc.pdf')
    plot_kde(df, 'mse', "all").savefig(results_filepath + 'filter_all_mse_kde.pdf')
    plot_hist(df, 'mse', "all").savefig(results_filepath + 'filter_all_mse_hist.pdf')
    plot_roc(df, "scan").savefig(results_filepath + 'filter_scan_auc-roc.pdf')
    plot_kde(df, 'mse', "scan").savefig(results_filepath + 'filter_scan_mse_kde.pdf')
    plot_hist(df, 'mse', "scan").savefig(results_filepath + 'filter_scan_mse_hist.pdf')
    plt.close('all')
    for anomaly_label in df[~df['label'].str.contains('background')].label.unique():
        plot_roc(df, anomaly_label).savefig(results_filepath + f'filter_{anomaly_label}_auc-roc.pdf')
        plot_kde(df, 'mse', anomaly_label).savefig(results_filepath + f'filter_{anomaly_label}_mse_kde.pdf')
        plot_hist(df, 'mse', anomaly_label).savefig(results_filepath + f'filter_{anomaly_label}_mse_hist.pdf')
        plt.close('all')
    
    binary_labels = df['label'].copy().astype('object')
    binary_labels.loc[binary_labels == 'background'] = int(0)
    binary_labels.loc[binary_labels != 0] = int(1)
    
    binary_predictions = df['filter_pred'].copy()
    binary_predictions.loc[:] = int(0)
    binary_predictions.loc[df.mse_loss >= reconstruction_error_threshold] = int(1)
    
    logger.info('Reconstruction error threshold = ', reconstruction_error_threshold)
    logger.info('AUC:', metrics.roc_auc_score(list(binary_labels), list(binary_predictions)))
    logger.info(metrics.classification_report(list(binary_labels), list(binary_predictions), digits=6))
    logger.info(metrics.confusion_matrix(list(binary_labels), list(binary_predictions)))
    
    cmd = metrics.ConfusionMatrixDisplay(
        metrics.confusion_matrix(list(binary_labels), list(binary_predictions)),
        display_labels=['background','anomaly'])
    cmd.plot()
    cmd.figure_.savefig(results_filepath + 'filter_confusion-matrix.pdf')
    

if __name__ == '__main__':
    main()
