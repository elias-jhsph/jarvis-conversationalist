# Jarvis Conversationalist

Jarvis Conversationalist is a Python-based project that provides a conversational interface using OpenAI's GPT-4 model. It uses text-to-speech and speech-to-text functionalities to facilitate a more interactive and engaging user experience.

## Features

- Real-time text-to-speech conversion: The application converts text responses from the GPT-4 model into speech, providing an auditory response to the user.
- Speech-to-text conversion: The application can convert spoken user input into text, which is then processed by the GPT-4 model.
- Multiprocessing and threading: The application uses Python's multiprocessing and threading capabilities to handle multiple tasks concurrently, ensuring smooth and responsive operation.
- Configurable: The application allows users to set their OpenAI API username and key, and also provides an option to reset these settings.

## Installation

This package is availible here [https://pypi.org/project/jarvis-conversationalist/](https://pypi.org/project/jarvis-conversationalist/)

This project requires Python 3.11 or later. This app currently works on Mac and Linux (which can cuda accelerate if able):


```bash
pip install jarvis_conversationalist
```

## Usage

To start the application run this in your terminal after pip installing:

```bash
jarvis
```

You can interrupt the conversation by pressing "Enter". To quit the application, press "Esc" then "Enter".

## Configuration

You can set your OpenAI API username and key using the `--user` and `--key` command-line arguments, respectively:

```bash
jarvis --user YOUR_USERNAME --key YOUR_KEY
```

To reset the saved username and key, use the `--reset` command-line argument:

```bash
jarvis --reset
```

## Non Technical Installation Instructions

**Installing Jarvis Conversationalist on a Mac**\=

1\. Install Python:  
Jarvis Conversationalist requires Python 3.11 or later.  
Go to the Python official website and download the version 3.11 for Mac.
or just download this... 
[https://www.python.org/ftp/python/3.11.0/python-3.11.0-macos11.pkg](https://www.python.org/ftp/python/3.11.0/python-3.11.0-macos11.pkg)

Open the downloaded file and follow the installation instructions.  
  
2\. Install Jarvis Conversationalist:  
In Terminal, type:

```bash
pip3.11 install jarvis_conversational
```

Then press Enter.  
Wait for the installation to complete.
  
3\. Finding Your OpenAI API Key - Create Open an OpenAI Account:  
Visit OpenAI's website and sign up or log in. After logging in, navigate to the API section.
[https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)


4\. Generate an API Key:  
In the API section, you should see an option to create a new API key.  
Follow the instructions to generate a new key.  
Once generated, make sure to copy and save your API key in a secure place.  
Using Jarvis Conversationalist  

5\. Start the Application:  
  
To set your OpenAI API username and key, use:  
```bash
jarvis --user YOUR_USERNAME --key YOUR_KEY 
```
Replace YOUR_USERNAME and YOUR_KEY with your actual OpenAI username and API key.  
After your first successful use go to terminal

```bash
jarvis
```

Then press Enter, to use Jarvis!  

6\. Updating Jarvis Conversationalist:

To update Jarvis Conversationalist, use:  
```bash
pip3.11 install --upgrade jarvis_conversationalist 
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the GPL 3 license.
