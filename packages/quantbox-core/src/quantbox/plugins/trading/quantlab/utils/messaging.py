"""
Notification and messaging system for sending alerts and communications
"""

# --- Imports ---
#%%
import os
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
from typing import Any
from .logging_utils import get_logger

logger = get_logger()

# --- Email Messaging ---
#%%
def send_email(sender, receivers, subject, body=None, body_html=None, attachments=None, smtp_server_name='etfi1003.bphtfi.pl'):
    """
    Sends an email with optional plain text or HTML body and attachments.

    :param sender: Email address of the sender.
    :param receivers: List of recipient email addresses.
    :param subject: Email subject.
    :param body: Plain text email body (optional).
    :param body_html: HTML email body (optional, used if `body` is None).
    :param attachments: List of file paths to attach (optional).
    :param smtp_server_name: SMTP server address.
    """
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ';'.join(receivers)
    msg['Subject'] = subject

    # Attach body (plain text or HTML)
    email_body = MIMEText(textwrap.dedent(body).strip(), 'plain') if body else MIMEText(body_html, 'html')
    msg.attach(email_body)

    # Attach files if any
    if attachments:
        for file_path in attachments:
            try:
                with open(file_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={Path(file_path).name}')
                    msg.attach(part)
            except FileNotFoundError:
                logger.warning(f"Attachment {file_path} not found, skipping.")

    # Send email
    try:
        with smtplib.SMTP(smtp_server_name) as smtp:
            smtp.sendmail(sender, receivers, msg.as_string())
        logger.info("Successfully sent email to %s", receivers)
    except smtplib.SMTPException as e:
        logger.error("Error sending email: %s", str(e))
        
# --- Slack Messaging ---
#%%
MAX_SLACK_TEXT_LENGTH = 3000

def split_long_text(text, max_length=MAX_SLACK_TEXT_LENGTH):
    """Splits long text into smaller chunks of max_length while keeping markdown formatting."""
    chunks = []
    while len(text) > max_length:
        split_index = text.rfind("\n", 0, max_length)  # Try to split at a new line
        if split_index == -1:  # If no new line found, force a hard split
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].strip()  # Remove leading/trailing spaces from the next chunk
    chunks.append(text)  # Add the remaining text
    return chunks

#%%
def process_slack_blocks(blocks):
    """Loops through blocks and splits long text into multiple sections if needed."""
    new_blocks = []
    for block in blocks:
        if block.get("type") == "section" and "text" in block:
            original_text = block["text"]["text"]
            if len(original_text) > MAX_SLACK_TEXT_LENGTH:
                split_texts = split_long_text(original_text)
                for text_chunk in split_texts:
                    new_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text_chunk}})
            else:
                new_blocks.append(block)
        else:
            new_blocks.append(block)  # Keep non-text blocks unchanged
    return new_blocks

#%%
def send_slack_message(subject, body, attachments, config_slack, create_body_blocks=True):
    # get config
    slack_token = config_slack['slack_bot_token']
    channel_id = config_slack['channel_id']
    client = WebClient(token=slack_token)

    # Create message blocks
    if create_body_blocks:
        message_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{subject}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{body}"
                }
            }
        ]
    else:
        # Create message blocks without markdown
        message_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": subject
                }
            }]
        message_blocks += body
        
    message_blocks = process_slack_blocks(message_blocks)
        
    try:
        # Send the message 
        client.chat_postMessage(
            channel=channel_id,
            text="Pipeline Summary",
            blocks=message_blocks
        )
        logger.info(f"Successfully sent Slack message to channel {channel_id}")
    except SlackApiError as e:
        logger.error(f"Error sending message to Slack: {e.response['error']}")
        return
    except Exception as e:
        logger.error(f"Unexpected error sending Slack message: {str(e)}")
        return

    file_permalinks = []
    for attachment in attachments:
        if os.path.exists(attachment):
            try:
                upload_response = client.files_upload_v2(
                    file=attachment,
                    channel=channel_id,
                    initial_comment=f"Log file for {subject}"
                )
                file_permalinks.append(upload_response["file"]["permalink"])
                logger.info(f"Uploaded file {attachment} to Slack channel {channel_id}")
            except SlackApiError as e:
                logger.error(f"Error uploading file {attachment} to Slack: {e.response['error']}")
            except Exception as e:
                logger.error(f"Unexpected error uploading file {attachment} to Slack: {str(e)}")
        else:
            logger.warning(f"Attachment {attachment} not found, skipping Slack upload.")

# --- Telegram Messaging ---
#%%
def send_telegram_message(token: str, chat_id: str, message: str, parse_mode: str = 'HTML') -> Any:
    """Send a message via Telegram.
    Args:
        token: Telegram bot token.
        chat_id: Target chat ID.
        message: Message to send.
        parse_mode: Telegram parse mode ('HTML', 'Markdown', or 'MarkdownV2'). Defaults to 'HTML'.
    Returns:
        Telegram API response as dict.
    """
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': parse_mode
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logger.info(f"Successfully sent Telegram message to chat_id {chat_id}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error sending Telegram message: {str(e)}")
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram message: {str(e)}")
        return {"ok": False, "error": str(e)}

#%%
def send_telegram_photo(token: str, chat_id: str, photo_path: str) -> Any:
    """Send a photo to a Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            data = {'chat_id': chat_id}
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            logger.info(f"Successfully sent photo {photo_path} to Telegram chat_id {chat_id}")
            return response.json()
    except FileNotFoundError:
        logger.error(f"Photo file {photo_path} not found for Telegram upload.")
        return {"ok": False, "error": f"File not found: {photo_path}"}
    except requests.RequestException as e:
        logger.error(f"Error sending photo to Telegram: {str(e)}")
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error sending photo to Telegram: {str(e)}")
        return {"ok": False, "error": str(e)}

#%%
def send_telegram_document(token: str, chat_id: str, document_path: str, caption: str = None) -> Any:
    """Send a document to a Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(document_path, 'rb') as document_file:
            files = {'document': document_file}
            data = {'chat_id': chat_id}
            if caption:
                data['caption'] = caption
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            logger.info(f"Successfully sent document {document_path} to Telegram chat_id {chat_id}")
            return response.json()
    except FileNotFoundError:
        logger.error(f"Document file {document_path} not found for Telegram upload.")
        return {"ok": False, "error": f"File not found: {document_path}"}
    except requests.RequestException as e:
        logger.error(f"Error sending document to Telegram: {str(e)}")
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error sending document to Telegram: {str(e)}")
        return {"ok": False, "error": str(e)}