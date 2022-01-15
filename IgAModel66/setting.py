import platform
import yaml
import os

print()
print('load config')


curDir=os.path.dirname(__file__)  # 当前文件夹
if platform.system().lower() == 'windows':
    print("windows")
    f=open(curDir+'/config.yaml', encoding='utf-8')
    config=yaml.safe_load(f)
elif platform.system().lower() == 'linux':
    print("linux")
    f=open(curDir+'/config.yaml', encoding='utf-8')
    config=yaml.safe_load(f)
    result=os.popen('echo "$USER"')
    user=result.read().strip()
    config.update(config['user'][user])
    
# config['server_path']=config['user'][user]['server_path']
print(config)
print()