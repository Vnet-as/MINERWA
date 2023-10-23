# Ansible scheudled attack generation in traffic

This ansible playbooks are leveraged to generate attacks into traffic according to predefined scenario
```
                                                                                                  
                                                                                                  
     +----------+                                                                                 
     | attack   |                                                                                 
     | scenario |                                                                                 
     |          |                     +---------------------+                                     
     |          |                     |                     |      attacks                        
     +----------+                    >|     attacker 1      |------------------------+            
           |               ssh     -/ |                     |                        |            
           |                     -/   +---------------------+                        |            
           |                   -/                                                    |            
+----------v-----------+     -/       +---------------------+            +-----------v-----------+
|                      |   -/         |                     |            |                       |
|  ansible C&C server  | -/     ----->|     attacker 2      |------------+        victim         |
|                      |/------/      |                     |            |                       |
|                      |-\            +---------------------+            +-----------^-----------+
+----------------------+  ---\                                                       |            
                              --\     +---------------------+                        |            
                                 ---\ |                     |                        |            
                                     ->      attacker N     |------------------------+            
                                      |                     |                                     
                                      +---------------------+                                     
                                                                                                  
```
Ansible playbooks execute attack from pre-configured attack schedule (week schedule) on victim
which represents simple web server.

Whole ifrastructure was spawned in multiple cloud environments (attackers and victim are separated). We used ubuntu 20 and kali 6 os images

Example inventory.cfg for this playbooks 
```
victim ansible_host=192.168.0.10 ansible_user=minerwa ansible_pass="pass" ansible_ssh_private_key_file="/path/to/private/key"

attacker1 ansible_host=192.168.0.11 ansible_user=minerwa ansible_pass="pass"
attacker2 ansible_host=192.168.0.12 ansible_user=minerwa ansible_pass="pass"
attacker3 ansible_host=192.168.0.13 ansible_user=minerwa ansible_pass="pass"

[attackers]
attacker1
attacker2
attacker3
```

Playbooks description:

run_attack_scenario.yaml - this playbooks executes attacks according to array of attacks defined in yaml format. Refer to vars/scenarios/test_scenario.yaml. Scenarios are included with `vars_files` in playbook and variable ATTACK_SCENARIO in playbook points to included scenario. 
ddos_attack.yaml - executed from execute_attack playbook. Realizes ddos attacks from all attackers
execute_attack.yaml - assembles command from catalog of commands (./vars/attacks/attacks_pool.yaml), selects attacker and executes command. executed from run_attack_scenario.yaml
init_attacker.yaml - inits attacker hosts, for example transfers some basic attack dictionaries
init_victim.yaml - sets up simple wordpress web server
print_schedule.yaml - prints attack schedule into output dir. Executed from run_attack_scenario.yaml
