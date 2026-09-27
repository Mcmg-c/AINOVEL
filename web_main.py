# -*- coding: utf-8 -*-
import threading
import webbrowser

import uvicorn


def main():
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
