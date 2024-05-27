1. Connect to a compute node using Slurm:

   ```bash
   srun -N 1 -p <partition> --pty bash
   ```

2. Download the Visual Studio Code CLI for Linux:

   ```bash
   curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
   ```

3. Extract the downloaded file:

   ```bash
   tar -xf vscode_cli.tar.gz
   ```

4. Run the Visual Studio Code tunneling tool:

   ```bash
   ./code tunnel
   ```

5. Follow the instructions provided by the tunneling tool.

6. In your local Visual Studio Code instance, navigate to the bottom left corner and select "Connect to a Tunnel".
