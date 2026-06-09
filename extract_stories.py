
import os
import pandas as pd
import requests
import re
from crime_fiction import exponential_backoff

def get_book_data(num_books=None):
    # Specify the path to your CSV file
    csv_file = os.environ.get("DETECTIVE_FICTION_CSV", "data/detective-fiction-PG.csv")
    # Use pandas to read the CSV file
    data = pd.read_csv(csv_file)

    book_nums = data["Text#"]
    titles = data["Title"]
    authors = data["Authors"]
    if num_books is None:
        return book_nums, titles, authors
    return book_nums[:num_books], titles[:num_books], authors[:num_books]

def download_book(book_id, save_file=False):
    url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    response = requests.get(url)
    text = response.text

    # Remove metadata section at the beginning of the text
    start_index = text.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    text = text[start_index:]
    text = text[text.find('\n') + 1:].strip()
    # Check if the book has multiple short stories
    if '*** END OF THE PROJECT GUTENBERG EBOOK' in text:
        story_divider = '*** END OF '
    elif '*** END OF THIS PROJECT GUTENBERG EBOOK' in text:
        story_divider = '*** END OF THIS'
    else:
        story_divider = None

    if story_divider:
        # Divide the book into stories
        story = text.split(story_divider)[0]
        # for i, story in enumerate(stories):
            # Remove extra newlines and leading/trailing whitespaces
        story = story.strip()
        story = re.sub(r'\n+', '\n', story)

        # if save_file:
        #     # Save each story as a separate text file
        #     with open(f'story_{i + 1}.txt', 'w', encoding='utf-8') as file:
        #         file.write(story)

    return story


def make_book_corpus(num_books=None):
    # Example usage
    num_books = None  # Number of books to fetch
    book_ids, titles, authors = get_book_data(num_books)
    book_dict = {book_id: {"title": title,
                           "author": author,
                           "text": download_book(book_id)}
                 for book_id, title, author in zip(book_ids, titles, authors)}
    return book_dict

def load_book_corpus():
    import json
    # with open(os.environ.get("DETECTIVE_BOOKS_JSON", "data/detective-books.json"), 'w') as file:
    #     json.dump(book_dict, file)
    with open(os.environ.get("DETECTIVE_BOOKS_JSON", "data/detective-books.json"), 'r') as file:
        book_dict = json.load(file)
    print("Done")
    return book_dict

# ************************************

def is_stories(client=None, text=""):
    m1 = """
    I'll give you the beginning of a book for the Gutenberg Project. 
    I want you to answer whether is seems that the book has a collection of stories or a single story.
    Your answer should be a single word - "collection" or "single".

    The beginning of the book:
    """
    m2 = """
    Now, I want you to give me a python list with the names of the stories. 
    Give me a list only.
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": m1})
    completion = exponential_backoff(client, messages)
    content = completion.choices[0].message.content
    messages.append({"role": completion.choices[0].message.role, "content": content})
    names = []

    if completion.choices[0].message.content == "collection":
        messages.append({"role": "user", "content": m2})
        completion = exponential_backoff(client, messages)
        content = completion.choices[0].message.content
        messages.append({"role": completion.choices[0].message.role, "content": content})
        names = content.split("[")[1].split("]")[0].replace("\"", "").replace("\'", "").split(",")

    return content, names


def make_stories():
    from openai import OpenAI
    from openai_client import get_client
    client = get_client()

    books = load_book_corpus()
    openings = {k: v["text"][:2000] for k, v in books.items()}
    stories_dict = {}
    for k, v in openings:
        stories_dict[k] = is_stories(client, v)