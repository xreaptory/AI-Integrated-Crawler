"""Tkinter user interface for the crawler."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import webbrowser
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .core import AIDrivenCrawler, PageSummary

BG_COLOR = "#F7F7F7"
FG_COLOR = "#0D0D0D"
ACCENT_COLOR = "#101820"
ACCENT_LIGHT = "#1E2A38"
HIGHLIGHT_COLOR = "#FFFFFF"
DEFAULT_MAX_PAGES = 25
DEFAULT_MAX_DEPTH = 2
DEFAULT_DELAY = 1.0


class QueueHandler(logging.Handler):
    """Route log messages into a thread-safe queue for the GUI."""

    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = output_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback
            self.handleError(record)
            return
        self._queue.put(message)


class CrawlerGUI:
    """Tkinter interface that runs the crawler in a background thread."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Web Crawler")
        self.root.geometry("980x720")
        self.root.minsize(860, 620)
        self.root.configure(bg=BG_COLOR)

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure(
            "Accent.TButton",
            background=ACCENT_COLOR,
            foreground=HIGHLIGHT_COLOR,
            padding=10,
            font=("Segoe UI", 11, "bold"),
        )
        self.style.map(
            "Accent.TButton",
            background=[("pressed", ACCENT_LIGHT), ("active", ACCENT_LIGHT)],
            foreground=[("disabled", "#A0A0A0")],
        )
        self.style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=HIGHLIGHT_COLOR,
            background=ACCENT_COLOR,
            lightcolor=ACCENT_COLOR,
            darkcolor=ACCENT_LIGHT,
            bordercolor=HIGHLIGHT_COLOR,
        )
        self.style.configure(
            "Treeview",
            background=HIGHLIGHT_COLOR,
            fieldbackground=HIGHLIGHT_COLOR,
            foreground=FG_COLOR,
            bordercolor=ACCENT_COLOR,
            rowheight=26,
        )
        self.style.map(
            "Treeview",
            background=[("selected", "#E6EBF5")],
            foreground=[("selected", ACCENT_COLOR)],
        )

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = QueueHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.url_var = tk.StringVar()
        self.goal_var = tk.StringVar()
        self.max_pages_var = tk.IntVar(value=DEFAULT_MAX_PAGES)
        self.max_depth_var = tk.IntVar(value=DEFAULT_MAX_DEPTH)
        self.delay_var = tk.DoubleVar(value=DEFAULT_DELAY)
        self.allow_offsite_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Idle.")

        self.crawl_thread: Optional[threading.Thread] = None
        self._latest_results: List[PageSummary] = []
        self._displayed_results: List[PageSummary] = []
        self._is_crawling = False
        self._shine_offset = -220
        self._status_toggle = False
        self._progress_visible = False
        self.summary_link_var = tk.StringVar(value="No page selected")
        self._current_selection_url = ""
        self._goal_required_terms: List[str] = []
        self._goal_min_match = 0

        self._build_layout()
        self._poll_log_queue()
        self._animate_banner()
        self._pulse_status()

    def _build_layout(self) -> None:
        self.banner = tk.Canvas(self.root, height=120, bg=BG_COLOR, highlightthickness=0, bd=0)
        self.banner.pack(fill=tk.X, padx=0, pady=(18, 6))
        self._banner_base = self.banner.create_rectangle(40, 18, self.root.winfo_width() - 40, 110, fill=ACCENT_COLOR, outline="")
        self._banner_shine = self.banner.create_rectangle(-220, 18, -80, 110, fill=HIGHLIGHT_COLOR, outline="", stipple="gray25")
        self._banner_title = self.banner.create_text(70, 50, anchor="w", text="AI Web Crawler", font=("Segoe UI", 26, "bold"), fill=HIGHLIGHT_COLOR)
        self._banner_subtitle = self.banner.create_text(
            70,
            86,
            anchor="w",
            text="Discover relevant pages with AI-guided prioritisation and live insights.",
            font=("Segoe UI", 11),
            fill="#E6E6E6",
        )
        self.banner.bind("<Configure>", self._update_banner_layout)

        form_card = tk.Frame(self.root, bg=HIGHLIGHT_COLOR, highlightbackground=ACCENT_COLOR, highlightthickness=1, bd=0)
        form_card.pack(fill=tk.X, padx=24, pady=(0, 14))
        form = tk.Frame(form_card, bg=HIGHLIGHT_COLOR)
        form.pack(fill=tk.X, padx=20, pady=18)

        tk.Label(form, text="Crawl Configuration", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(form, text="Start URL", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=1, column=0, columnspan=4, sticky="w", pady=(16, 4))
        self.url_entry = tk.Entry(
            form,
            textvariable=self.url_var,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.url_entry.grid(row=2, column=0, columnspan=4, sticky="ew")

        tk.Label(form, text="Goal (optional)", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, columnspan=4, sticky="w", pady=(16, 4))
        self.goal_entry = tk.Entry(
            form,
            textvariable=self.goal_var,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.goal_entry.grid(row=4, column=0, columnspan=4, sticky="ew")

        controls = tk.Frame(form, bg=HIGHLIGHT_COLOR)
        controls.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(16, 0))
        controls.grid_columnconfigure(6, weight=1)

        tk.Label(controls, text="Max pages", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.pages_spin = tk.Spinbox(
            controls,
            from_=1,
            to=500,
            textvariable=self.max_pages_var,
            width=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.pages_spin.grid(row=0, column=1, padx=(8, 24))

        tk.Label(controls, text="Max depth", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w")
        self.depth_spin = tk.Spinbox(
            controls,
            from_=1,
            to=10,
            textvariable=self.max_depth_var,
            width=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.depth_spin.grid(row=0, column=3, padx=(8, 24))

        tk.Label(controls, text="Delay (s)", bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10, "bold")).grid(row=0, column=4, sticky="w")
        self.delay_spin = tk.Spinbox(
            controls,
            from_=0.0,
            to=10.0,
            increment=0.1,
            textvariable=self.delay_var,
            width=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            highlightbackground=ACCENT_COLOR,
            highlightcolor=ACCENT_COLOR,
            relief="flat",
            bd=1,
        )
        self.delay_spin.grid(row=0, column=5, padx=(8, 24))

        self.allow_offsite_check = tk.Checkbutton(
            controls,
            text="Allow offsite links",
            variable=self.allow_offsite_var,
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            selectcolor=HIGHLIGHT_COLOR,
            activebackground=HIGHLIGHT_COLOR,
            activeforeground=FG_COLOR,
            font=("Segoe UI", 10),
            highlightthickness=0,
        )
        self.allow_offsite_check.grid(row=0, column=6, padx=(0, 24))

        actions = tk.Frame(form_card, bg=HIGHLIGHT_COLOR)
        actions.pack(fill=tk.X, padx=20, pady=(0, 18))

        self.start_button = tk.Button(
            actions,
            text="Start crawl",
            command=self.start_crawl,
            bg=ACCENT_COLOR,
            fg=HIGHLIGHT_COLOR,
            activebackground=ACCENT_LIGHT,
            activeforeground=HIGHLIGHT_COLOR,
            font=("Segoe UI", 11, "bold"),
            padx=24,
            pady=10,
            borderwidth=0,
            relief="flat",
            cursor="hand2",
            disabledforeground="#8A8A8A",
        )
        self.start_button.pack(side=tk.LEFT)
        self.start_button.bind("<Enter>", lambda _: self._highlight_start(True))
        self.start_button.bind("<Leave>", lambda _: self._highlight_start(False))

        self.progress = ttk.Progressbar(actions, mode="indeterminate", style="Accent.Horizontal.TProgressbar", length=180)

        status_frame = tk.Frame(actions, bg=HIGHLIGHT_COLOR)
        status_frame.pack(side=tk.RIGHT)
        self.status_canvas = tk.Canvas(status_frame, width=18, height=18, bg=HIGHLIGHT_COLOR, highlightthickness=0, bd=0)
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 8))
        self._status_dot = self.status_canvas.create_oval(4, 4, 14, 14, fill="#B5B5B5", outline="")
        tk.Label(status_frame, textvariable=self.status_var, bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10)).pack(side=tk.LEFT)

        content = tk.Frame(self.root, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))

        results_card = tk.Frame(content, bg=HIGHLIGHT_COLOR, highlightbackground=ACCENT_COLOR, highlightthickness=1, bd=0)
        results_card.pack(fill=tk.BOTH, expand=True, pady=(0, 14))
        tk.Label(results_card, text="Collected Pages", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=20, pady=(16, 6))

        tree_container = tk.Frame(results_card, bg=HIGHLIGHT_COLOR)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=20)
        columns = ("title", "score", "url")
        self.results_tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="browse")
        self.results_tree.heading("title", text="Title")
        self.results_tree.heading("score", text="Relevance")
        self.results_tree.heading("url", text="URL")
        self.results_tree.column("title", width=320, anchor="w")
        self.results_tree.column("score", width=120, anchor="center")
        self.results_tree.column("url", width=420, anchor="w")
        vsb = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=vsb.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        self.results_tree.tag_configure("odd", background="#F0F2F5")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_selected)
        self.results_tree.bind("<Double-1>", self._on_result_activated)
        self.results_tree.bind("<Return>", self._on_result_activated)

        tk.Label(results_card, text="Selected Page Summary", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(12, 4))
        link_row = tk.Frame(results_card, bg=HIGHLIGHT_COLOR)
        link_row.pack(fill=tk.X, padx=20, pady=(0, 6))
        self.summary_link = tk.Label(
            link_row,
            textvariable=self.summary_link_var,
            bg=HIGHLIGHT_COLOR,
            fg="#808080",
            font=("Segoe UI", 10, "underline"),
            cursor="arrow",
        )
        self.summary_link.pack(side=tk.LEFT)
        self.summary_link.bind("<Button-1>", self._open_selected_url)
        self.open_button = tk.Button(
            link_row,
            text="Open in browser ↗",
            command=self._open_selected_url,
            bg=ACCENT_COLOR,
            fg=HIGHLIGHT_COLOR,
            activebackground=ACCENT_LIGHT,
            activeforeground=HIGHLIGHT_COLOR,
            padx=12,
            pady=4,
            borderwidth=0,
            relief="flat",
            cursor="arrow",
            state="disabled",
        )
        self.open_button.pack(side=tk.RIGHT)
        self.summary_text = ScrolledText(
            results_card,
            height=6,
            font=("Segoe UI", 10),
            bg=HIGHLIGHT_COLOR,
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
            wrap=tk.WORD,
        )
        self.summary_text.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 16))
        self.summary_text.configure(borderwidth=0, highlightthickness=1, highlightbackground=ACCENT_COLOR)
        self.summary_text.tag_config("summary_title", font=("Segoe UI", 12, "bold"))
        self.summary_text.tag_config("hyperlink", foreground=ACCENT_COLOR)

        log_card = tk.Frame(content, bg=HIGHLIGHT_COLOR, highlightbackground=ACCENT_COLOR, highlightthickness=1, bd=0)
        log_card.pack(fill=tk.BOTH, expand=True)
        tk.Label(log_card, text="Crawler Log", bg=HIGHLIGHT_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(16, 6))
        self.log_text = ScrolledText(
            log_card,
            height=10,
            font=("Consolas", 10),
            bg="#11151C",
            fg="#E7E9ED",
            insertbackground="#E7E9ED",
            wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 16))
        self.log_text.configure(state="disabled")

        export_bar = tk.Frame(log_card, bg=HIGHLIGHT_COLOR)
        export_bar.pack(fill=tk.X, padx=20, pady=(0, 16))
        ttk.Button(export_bar, text="Export results", command=self.export_results, style="Accent.TButton").pack(side=tk.RIGHT)

        status_row = tk.Frame(log_card, bg=HIGHLIGHT_COLOR)
        status_row.pack(fill=tk.X, padx=20, pady=(0, 10))
        status_indicator = tk.Label(status_row, text="●", fg="#2B8A3E", bg=HIGHLIGHT_COLOR, font=("Segoe UI", 12))
        status_indicator.pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self.status_var, bg=HIGHLIGHT_COLOR, fg=FG_COLOR, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(6, 0))

    def _update_banner_layout(self, _event: tk.Event) -> None:
        width = max(self.root.winfo_width(), 280)
        self.banner.coords(self._banner_base, 40, 18, width - 40, 110)

    def _animate_banner(self) -> None:
        width = max(self.banner.winfo_width(), 200)
        self._shine_offset += 8
        if self._shine_offset > width + 120:
            self._shine_offset = -220
        self.banner.coords(self._banner_shine, self._shine_offset, 18, self._shine_offset + 160, 110)
        self.root.after(50, self._animate_banner)

    def _pulse_status(self) -> None:
        if self._is_crawling:
            self._status_toggle = not self._status_toggle
            fill = ACCENT_COLOR if self._status_toggle else ACCENT_LIGHT
        else:
            self._status_toggle = False
            fill = "#B5B5B5"
        self.status_canvas.itemconfig(self._status_dot, fill=fill)
        self.root.after(320, self._pulse_status)

    def _highlight_start(self, hover: bool) -> None:
        if self._is_crawling:
            return
        self.start_button.configure(bg=ACCENT_LIGHT if hover else ACCENT_COLOR)

    def _poll_log_queue(self) -> None:
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_output(message)
        self.root.after(150, self._poll_log_queue)

    def _append_output(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_busy_state(self, running: bool) -> None:
        self._is_crawling = running
        if running:
            self.start_button.configure(state="disabled", text="Crawling...", bg=ACCENT_LIGHT)
            if not self._progress_visible:
                self.progress.pack(side=tk.LEFT, padx=(20, 0))
                self._progress_visible = True
            self.progress.start(12)
        else:
            self.start_button.configure(state="normal", text="Start crawl", bg=ACCENT_COLOR)
            self.progress.stop()
            if self._progress_visible:
                self.progress.pack_forget()
                self._progress_visible = False

    def _set_summary_text(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.config(cursor="arrow")

    def _set_summary_link(self, url: Optional[str]) -> None:
        if url:
            display = url if len(url) <= 70 else url[:67] + "..."
            self.summary_link_var.set(display)
            self.summary_link.configure(fg=ACCENT_COLOR, cursor="hand2")
            self._current_selection_url = url
            self.open_button.configure(state="normal", cursor="hand2", bg=ACCENT_COLOR, fg=HIGHLIGHT_COLOR)
        else:
            self.summary_link_var.set("No page selected")
            self.summary_link.configure(fg="#808080", cursor="arrow")
            self._current_selection_url = ""
            self.open_button.configure(state="disabled", cursor="arrow", bg="#D8D8D8", fg="#6F6F6F")

    def _open_url(self, url: str) -> None:
        if not url:
            return
        try:
            logging.info("Opening in browser: %s", url)
            webbrowser.open(url)
        except Exception as error:  # pragma: no cover - platform browser failure
            logging.warning("Unable to open URL %s: %s", url, error)

    def _open_selected_url(self, _event: Optional[tk.Event] = None) -> None:
        self._open_url(self._current_selection_url)

    def _on_result_activated(self, _event: tk.Event) -> str:
        selection = self.results_tree.selection()
        if not selection:
            return "break"
        index = int(selection[0])
        if 0 <= index < len(self._displayed_results):
            page = self._displayed_results[index]
            self._open_url(page.url)
        return "break"

    def _clear_results(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._displayed_results = []
        self._goal_required_terms = []
        self._goal_min_match = 0
        self._set_summary_text("Crawl running... results will show here.")
        self._set_summary_link(None)

    def start_crawl(self) -> None:
        if self.crawl_thread and self.crawl_thread.is_alive():
            messagebox.showinfo("Crawler", "A crawl is already in progress.")
            return

        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Missing URL", "Please enter a starting URL.")
            return

        goal = self.goal_var.get().strip() or None
        try:
            max_pages = max(1, int(self.max_pages_var.get()))
            max_depth = max(1, int(self.max_depth_var.get()))
            delay = max(0.0, float(self.delay_var.get()))
        except ValueError:
            messagebox.showerror("Invalid input", "Please check the crawl settings.")
            return

        allow_offsite = self.allow_offsite_var.get()

        self._clear_results()
        self.status_var.set("Crawling...")
        self._set_busy_state(True)
        self._append_output(f"Starting crawl at {url}")

        self.crawl_thread = threading.Thread(
            target=self._run_crawler,
            args=(url, goal, max_pages, max_depth, delay, allow_offsite),
            daemon=True,
        )
        self.crawl_thread.start()

    def _run_crawler(
        self,
        url: str,
        goal: Optional[str],
        max_pages: int,
        max_depth: int,
        delay: float,
        allow_offsite: bool,
    ) -> None:
        try:
            crawler = AIDrivenCrawler(
                root_url=url,
                query=goal,
                max_pages=max_pages,
                max_depth=max_depth,
                delay=delay,
                stay_in_domain=not allow_offsite,
            )
            crawler.run()
        except Exception as exc:  # pragma: no cover - surfaced via GUI
            logging.exception("Crawler failed: %s", exc)
            self.root.after(0, lambda: self._on_crawl_failed(exc))
            return

        self.root.after(0, lambda: self._on_crawl_finished(crawler))

    def _populate_results(self, results: List[PageSummary]) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._displayed_results = results
        self._set_summary_link(None)
        if not results:
            message = "No pages were collected. Adjust crawl limits and try again."
            self._set_summary_text(message)
            return
        for index, page in enumerate(results):
            tag = "odd" if index % 2 else "even"
            self.results_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(page.title, f"{page.score:.3f}", page.url),
                tags=(tag,),
            )
        self.results_tree.tag_configure("even", background=HIGHLIGHT_COLOR)
        first = self.results_tree.get_children()
        if first:
            self.results_tree.selection_set(first[0])
            self._show_result_details(results[0])

    def _show_result_details(self, page: PageSummary) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, page.title + "\n", ("summary_title",))
        self.summary_text.insert(tk.END, page.url + "\n", ("hyperlink",))
        self.summary_text.insert(tk.END, "\nRelevance score: {0:.3f}\n".format(page.score))
        if page.keywords:
            self.summary_text.insert(tk.END, "Keywords: {0}\n".format(", ".join(page.keywords)))
        if page.summary_text:
            self.summary_text.insert(tk.END, "\nSummary:\n")
            self.summary_text.insert(tk.END, page.summary_text.strip() + "\n")
        if page.summary_sentences:
            self.summary_text.insert(tk.END, "\nKey sentences:\n")
            for sentence in page.summary_sentences:
                self.summary_text.insert(tk.END, f"- {sentence}\n")
        if page.goal_required_terms:
            status = "met" if page.goal_match_passed else "partial"
            needed = max(self._goal_min_match, 1)
            total = len(page.goal_required_terms)
            matched_terms = ", ".join(page.goal_match_terms) if page.goal_match_terms else "none"
            missing_terms = ", ".join(page.goal_missing_terms) if page.goal_missing_terms else "none"
            self.summary_text.insert(
                tk.END,
                "\nGoal heuristics ({status}, needs {needed}/{total} terms):\n".format(
                    status=status,
                    needed=needed,
                    total=total,
                ),
            )
            self.summary_text.insert(tk.END, f"  matched: {matched_terms}\n")
            if missing_terms != "none":
                self.summary_text.insert(tk.END, f"  remaining: {missing_terms}\n")
            if page.goal_must_terms:
                must_display = ", ".join(page.goal_must_terms)
                missing_must = (
                    ", ".join(page.goal_missing_must)
                    if page.goal_missing_must
                    else "none"
                )
                self.summary_text.insert(tk.END, f"  must-have terms: {must_display}\n")
                if missing_must != "none":
                    self.summary_text.insert(tk.END, f"  missing must-have: {missing_must}\n")
        self.summary_text.config(cursor="arrow")
        self._set_summary_link(page.url)

    def _on_result_selected(self, event: tk.Event) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self._displayed_results):
            self._show_result_details(self._displayed_results[index])

    def _on_crawl_finished(self, crawler: AIDrivenCrawler) -> None:
        self._set_busy_state(False)
        self._goal_required_terms = list(crawler.goal_required_terms)
        self._goal_min_match = crawler.goal_min_match
        results = sorted(crawler.results, key=lambda item: item.score, reverse=True)
        self._latest_results = results
        count = len(results)
        base_status = f"Finished. {count} page(s) collected."
        self.status_var.set(base_status)
        self._append_output(f"Finished crawl. {count} page(s) collected.")
        if self._goal_required_terms:
            terms_display = ", ".join(self._goal_required_terms)
            self._append_output(
                f"Goal heuristics tracked: needs {self._goal_min_match} of [{terms_display}] (informational)."
            )
        self._populate_results(results)

    def _on_crawl_failed(self, error: Exception) -> None:
        self._set_busy_state(False)
        self.status_var.set("Error while crawling.")
        messagebox.showerror("Crawler error", str(error))

    def export_results(self) -> None:
        if not self._latest_results:
            messagebox.showinfo("Crawler", "No results to export yet.")
            return
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [
                {
                    "title": page.title,
                    "url": page.url,
                    "score": page.score,
                    "keywords": page.keywords,
                    "summary": page.summary_sentences,
                    "summary_text": page.summary_text,
                }
                for page in self._latest_results
            ],
        }
        file_path = filedialog.asksaveasfilename(
            title="Save crawler results",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            messagebox.showerror("Export failed", f"Could not save file: {error}")
            return
        messagebox.showinfo("Crawler", "Results exported successfully.")

    def on_close(self) -> None:
        if self.crawl_thread and self.crawl_thread.is_alive():
            if not messagebox.askyesno("Quit", "A crawl is still running. Quit anyway?"):
                return
        logging.getLogger().removeHandler(self.log_handler)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_app() -> None:
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.INFO)
    app = CrawlerGUI()
    app.run()


__all__ = ["CrawlerGUI", "run_app"]
