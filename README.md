# hpdec - Human Presence Detection with CSI Data

Update soon
<!-- `hpdec` is a repository that enables real-time indoor human presence detection using Channel State Information (CSI) data obtained from a Raspberry Pi CSI extractor. 
This repository was created to evaluate pre-trained model with data collected by [CALS](https://github.com/INCLab/CALS), and despite being a shallow model, achieved approximately 94% accuracy at window size 70.
<p align="center"><img src="https://user-images.githubusercontent.com/51084152/231768156-e34982d7-57c4-49a2-b121-894b81fbac4a.png"  width="500" height="350"/></p>

## Prerequisites

 - Python 3.x
 - [Nexmon CSI Extractor](https://github.com/seemoo-lab/nexmon_csi) Raspberry Pi B3+/B4(Wi-Fi chip: bcm43455c0) 
 
---
## Installation

1. Clone this repository on CSI extractor and server:

```bash
git clone https://github.com/cheeseBG/hpdec.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the CSI extractor on your Raspberry Pi:

```bash
python3 client.py
```

2. Run the server script on the server-side:

```bash
cd server
python3 rtClassfier_server.py
```

3. The server will analyze the incoming CSI data and display the presence status (either `Presence Detected` or `No Presence Detected`) based on the data analysis.

## Configuration

You can customize the server and client configurations by modifying the `config.yaml` file.

Example:
```yaml
# Server
server_ip: '192.168.1.100'
server_port: 9010

# Client
client_ip: '192.168.0.191'
client_port: 9009
client_mac_address: 'dca6328e1dcb'
```

## Referenced Projects

This project takes inspiration from the following open-source project:
- **Nexmon**: The Nexmon project provides firmware patches for collecting CSI on Broadcom Wi-Fi chips. For more information about this project, please visit the [Nexmon GitHub repository](https://github.com/seemoo-lab/nexmon_csi). -->
