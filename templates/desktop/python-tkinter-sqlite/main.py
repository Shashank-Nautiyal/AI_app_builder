from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from database import add_item, delete_item, init_db, list_items


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Desktop SQLite App")
        self.geometry("720x460")
        self.configure(padx=16, pady=16)

        self.title_var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self.title_var)
        entry.pack(fill="x")
        entry.bind("<Return>", lambda _event: self.add_current_item())

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=10)
        ttk.Button(actions, text="Add", command=self.add_current_item).pack(side="left")
        ttk.Button(actions, text="Delete Selected", command=self.delete_selected_item).pack(side="left", padx=8)

        self.tree = ttk.Treeview(self, columns=("id", "title", "status"), show="headings")
        self.tree.heading("id", text="ID")
        self.tree.heading("title", text="Title")
        self.tree.heading("status", text="Status")
        self.tree.column("id", width=80, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in list_items():
            self.tree.insert("", "end", iid=str(row["id"]), values=(row["id"], row["title"], row["status"]))

    def add_current_item(self) -> None:
        title = self.title_var.get().strip()
        if not title:
            return
        add_item(title)
        self.title_var.set("")
        self.refresh()

    def delete_selected_item(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        delete_item(int(selected[0]))
        self.refresh()


if __name__ == "__main__":
    init_db()
    App().mainloop()
