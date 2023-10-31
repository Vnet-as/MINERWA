#!/bin/bash

task_start="$(date +%s)"

cd "/data/kinit/test_id2t"
filename="merged-0823-08:18:52-0823-08:28:52"
#filename="merged-0823-15:58:52-0823-16:08:52"
#${filename}_short.pcap
screen -S ID2Tinjection-recalculate-statistics ~/ID2T/id2t -i ${filename}.pcap -ry
screen -S ID2Tinjection-portscan ~/ID2T/id2t -i ${filename}.pcap -a PortscanAttack packets.per-second=100 -a PortscanAttack packets.per-second=10 port.open=80,8080 -ie -o ${filename}-portscan.pcap
screen -S ID2Tinjection-smbscan ~/ID2T/id2t -i ${filename}.pcap -a SMBScanAttack packets.per-second=100 -a SMBScanAttack packets.per-second=10 target.count=10 -ie -o ${filename}-smbscan.pcap
screen -S ID2Tinjection-ms17scan ~/ID2T/id2t -i ${filename}.pcap -a MS17ScanAttack packets.per-second=100 -a MS17ScanAttack packets.per-second=10 -ie -o ${filename}-ms17scan.pcap
screen -S ID2Tinjection-ddos ~/ID2T/id2t -i ${filename}.pcap -a DDoSAttack -a DDoSAttack port.dst=80,443 packets.per-second=100 attack.duration=500 -ie -o ${filename}-ddos.pcap
screen -S ID2Tinjection-smbloris ~/ID2T/id2t -i ${filename}.pcap -a SMBLorisAttack -a SMBLorisAttack packets.per-second=100 attack.duration=500 -ie -o ${filename}-smbloris.pcap
screen -S ID2Tinjection-ethernalblue ~/ID2T/id2t -i ${filename}.pcap -a EternalBlueExploit -a EternalBlueExploit port.dst=443 packets.per-second=10 -ie -o ${filename}-ethernalblue.pcap
screen -S ID2Tinjection-salitybotnet ~/ID2T/id2t -i ${filename}.pcap -a SalityBotnet -a SalityBotnet packets.per-second=10 -ie -o ${filename}-salitybotnet.pcap

task_end="$(date +%s)"
echo -e "${filename} attacks generation took $((task_end - task_start)) secs" > ${filename}-time.log