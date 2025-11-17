Gemini Discord Bot - Sprint 1 (Text-Only)

This is the foundational code for the Gemini Discord Bot. This sprint's goal is to get the bot online, connected to Discord, and responding to text commands via the Gemini API.

1. Ubuntu Server Prerequisites

Make sure your Ubuntu server has the core Python tools installed.

sudo apt-get update
sudo apt-get install -y python3.8-venv python3-pip ffmpeg libnacl-dev


(Note: ffmpeg and libnacl-dev aren't used in Sprint 1, but we will need them for Sprint 2 (voice). Let's install them now.)

2. Project Setup

Clone/Copy Files:
Get bot.py, requirements.txt, and .env.example into a new directory on your server (e.g., /home/malvin/gemini-bot).

mkdir ~/gemini-bot
cd ~/gemini-bot
# (Assuming you've copied or created the files in here)


Create a Python Virtual Environment:
It's best practice to keep project dependencies isolated.

python3 -m venv venv


Activate the Environment:
You'll need to do this every time you start a new terminal session to work on the bot.

source venv/bin/activate


(Your prompt should now show (venv))

Install Python Dependencies:
This will install discord.py, google-generativeai, and python-dotenv.

pip install -r requirements.txt


2.5. Git & GitHub Setup (Optional but Recommended)

Using Git (and GitHub) will make managing updates much easier, just as you said.

Create .gitignore:
A .gitignore file tells Git to ignore sensitive files (like .env) and bulky folders (like venv). A file named .gitignore has been provided for you. Make sure it's in your ~/gemini-bot directory.

Initialize Git:
In your project directory (~/gemini-bot), run:

git init
git add .
git commit -m "Sprint 1: Initial commit - Text-only bot"


Create a GitHub Repository:

Go to GitHub and create a new repository (e.g., gemini-discord-bot).

IMPORTANT: Make it a Private repository. This is critical to protect your API keys and bot logic.

Do not add a README, license, or .gitignore from the GitHub UI (we've already created them).

Link and Push:
GitHub will give you instructions on the new repo's page. They will look something like this (use your own URL):

git remote add origin [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
git branch -M main
git push -u origin main


Now, when we make changes in future sprints, you can just run git add ., git commit -m "Sprint 2 changes", and git push to save your work.

3. Configuration


Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X` in nano).

## 4\. Invite Your Bot to Your Server

1.  Go back to the [Discord Developer Portal](https://www.google.com/search?q=https://discord.com/developers/applications).
2.  Select your application.
3.  Go to "OAuth2" -> "URL Generator".
4.  **Scopes:** Select `bot` AND `applications.commands`.
    * `bot`: Adds the bot user to your server.
    * `applications.commands`: Allows the bot to create and respond to slash commands.
5.  **Bot Permissions:** Select:
    * `Read Messages/View Channels`
      * `Send Messages`
    * `Read Message History`
        *(We'll add `Connect` and `Speak` in the next sprint)*
6.  Copy the generated URL at the bottom, paste it into your browser, and invite the bot to your server.

## 5. Run the Bot

With your `(venv)` still active, run the Python script:

```bash
python3 bot.py


If all went well, you'll see "Logged in as [Your Bot Name]" in the console, and your bot will appear "Online" in Discord.

6. Test Sprint 1

In any text channel your bot can see, start typing /
You should see the ask command pop up.

Select it and type your question:
/ask What is the capital of Türkiye?

The bot should respond with an answer from Gemini.

To stop the bot, press Ctrl+C in the terminal.

For a more permanent, production-ready setup (Sprint 4+), we'll look into using systemd to run this as a background service. For now, this is perfect for development.