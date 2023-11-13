# Jarvis Conversationalist

Jarvis Conversationalist is a Python-based project that provides a conversational interface using OpenAI's GPT-4 model. It uses text-to-speech and speech-to-text functionalities to facilitate a more interactive and engaging user experience.

## Features

- Real-time text-to-speech conversion: The application converts text responses from the GPT-4 model into speech, providing an auditory response to the user.
- Speech-to-text conversion: The application can convert spoken user input into text, which is then processed by the GPT-4 model.
- Multiprocessing and threading: The application uses Python's multiprocessing and threading capabilities to handle multiple tasks concurrently, ensuring smooth and responsive operation.
- Configurable: The application allows users to set their OpenAI API username and key, and also provides an option to reset these settings.

## Installation

This project requires Python 3.11 or later. For mac, install portaudio and then pip install the package:

```bash
brew install portaudio
pip install jarvis_conversationalist
```

## Usage

To start the application, run the `cli.py` script from the command line:

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

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the GPL 3 license.
