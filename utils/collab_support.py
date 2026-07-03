
# Google Colab support functions

# March - 2023 and before


# function to handle copy files from google file servers
import os
from shutil import copyfile
import subprocess

def download_expand_tar(remote_path, file_name):
    """ 
    Util function to upload code from Google storage to collab local
    Copy tar.bz2 file name to remote folder and uncompress
    IMPORTANT: we use sparse -S option always. Check it to compress!
    """
    if not os.path.isfile(os.path.join(os.getcwd(), file_name)):
      print('Copying ', file_name, ' from remote server...')
      copyfile(os.path.join(remote_path, file_name), 
              os.path.join(os.getcwd(), file_name))
      subprocess.run(['ls'])
      file_type = file_name.split('.')[-1]
      if file_type == 'tar':
        command_option = '-xvSf'
      elif file_type == 'xz':
        command_option = '-xf'
      elif file_type == 'gz':
        command_option = '-xvzSf'
      elif file_type == 'bz2':
        command_option = '-xvjSf'
      print('Uncompressing ', file_name, ' with', command_option, ' ...')
      subprocess.run(['tar', command_option, file_name])
      subprocess.run(['ls'])
    else:
      print('File ', file_name, ' already downloaded and uncompressed')


def expand_tar(remote_path, file_name):
    """ 
    IMPORTANT: we use sparse -S option always. Check it to compress!
    """
      subprocess.run(['ls'])
      file_type = file_name.split('.')[-1]
      if file_type == 'tar':
        command_option = '-xvSf'
      elif file_type == 'xz':
        command_option = '-xf'
      elif file_type == 'gz':
        command_option = '-xvzSf'
      elif file_type == 'bz2':
        command_option = '-xvjSf'
      print('Uncompressing ', file_name, ' with', command_option, ' ...')
      subprocess.run(['tar', command_option, file_name])
      subprocess.run(['ls'])
    else:
      print('File ', file_name, ' already uncompressed')


def download_expand_tar_old(remote_path, file_name):
    """
    Util function to upload code from Google storage to collab local
    Copy tar.bz2 file name to remote folder and uncompress
    """
    if not os.path.isfile(os.path.join(os.getcwd(), file_name)):
      print('Copying ', file_name, ' from remote server...')
      copyfile(os.path.join(remote_path, file_name),
              os.path.join(os.getcwd(), file_name))
      subprocess.run(['ls'])
      print('Uncompressing ', file_name, ' ...')
      if file_name[-3:] == 'tar':
        subprocess.run(['tar', '-xvf', file_name])
      elif file_name[-3:] == 'bz2':
        subprocess.run(['tar', '-xvjSf', file_name])
      else:
        print('Unknown file extension')
      subprocess.run(['ls'])
    else:
      print('File ', file_name, ' already downloaded and uncompressed')


def finish_and_leave():
    # Finish everything
    from google.colab import runtime
    runtime.unassign()
