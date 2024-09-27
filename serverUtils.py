import paramiko
from scp import SCPClient
import os

def create_ssh_client(hostname, port, username, password):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=port, username=username, password=password)
        return ssh
    except Exception as e:
        print(f"Failed to create SSH connection: {e}")
        return None


def upload_file(ssh_client, local_path, remote_path):
    """Uploads a file to the server."""
    with SCPClient(ssh_client.get_transport()) as scp:
        scp.put(local_path, remote_path)
        print(f"File {local_path} uploaded to {remote_path}")

def download_file(ssh_client, remote_path, local_path):
    """Downloads a file from the server."""
    with SCPClient(ssh_client.get_transport()) as scp:
        scp.get(remote_path, local_path)
        print(f"File {remote_path} downloaded to {local_path}")

def upload_geo(local_path, remote_path, ssh_client=None):
    """Auto gets the username and password from the environment variables"""
    host = "geodigital"
    port = 22
    if ssh_client is None:
        ssh_client = create_ssh_client(host, port, os.getenv('SERVER_USERNAME'), os.getenv('SERVER_PASSWORD'))
    upload_file(ssh_client, local_path, remote_path)

def download_geo(local_path, remote_path, ssh_client=None):
    """Auto gets the username and password from the environment variables"""
    host = "geodigital"
    port = 22
    if ssh_client is None:
        ssh_client = create_ssh_client(host, port, os.getenv('SERVER_USERNAME'), os.getenv('SERVER_PASSWORD'))
    download_file(ssh_client, remote_path, local_path)
    

# Example usage
if __name__ == "__main__":
    hostname = 'geodigital.inf.ufrgs.br' #Obs: testei no portal da UFRGS
    port = 22  # Default SSH port
    
    """
    Save the username and password as environment variables
    Windows: set SERVER_USERNAME=your_username
    Linux/Unix: export SERVER_USERNAME=your_username
    """

    #Obs: por motivos de segurança botei como variaveis do sistema
    #mas aqui para testes e enfim bote ser colocado diretamente as strings
    #username = os.getenv('SERVER_USERNAME')
    #password = os.getenv('SERVER_PASSWORD')
    
    username = "user_vgt"
    password = "#9VGT_User9#"
    
    local_file_path = 'teste'
    remote_file_path = '//home//user_vgt//testando//teste 1.txt'

    # Create SSH client
    ssh_client = create_ssh_client(hostname, port, username, password)
    print(ssh_client)
    
    # Upload a file
    #upload_file(ssh_client, local_file_path, remote_file_path)

    # Download a file
    download_file(ssh_client, remote_file_path, local_file_path)

    # Close the SSH connection
    ssh_client.close()
