# https://medium.com/@simon.hawe/save-time-by-automating-ssh-and-scp-tasks-with-python-e149de606c7b

from scp import SCPClient


# SCPCLient takes a paramiko transport as its only argument
scp = SCPClient(ssh.get_transport())

scp.put('file_path_on_local_machine', 'file_path_on_remote_machine')
scp.get('file_path_on_remote_machine', 'file_path_on_local_machine')

scp.close()