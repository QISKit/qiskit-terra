#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Check and fixup mailmaps files against git log."""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from enum import IntEnum, unique
from pathlib import Path
from sys import stderr
from unicodedata import normalize

# Set up some globals for simplicity
color_output = True
quiet_mode = False  # Only report errors
problem_found = False  # Problems requiring user intervention were detected
changes_made = False  # .mailmap was updated
mailmap_path = Path()  # We'll set it correctly later
mailmap_body_offset = 0  # Offset from the .mailmap header into the body


@unique
class ExitCode(IntEnum):
    """Program exit codes enumeration."""

    NO_PROBLEM = 0
    CHANGES_MADE = 1
    MAILMAP_NOT_FOUND = 2
    MAILMAP_FORMAT = 3
    GIT_LOG = 4
    MAILMAP_NEEDS_MANUAL_CHANGE = 5


def mailmap_line(index: int) -> str:
    """Format a string to refer to a mailmap line in console messages."""
    return f"{mailmap_path}:{mailmap_body_offset+index}"


def print_red(text: str) -> None:
    """Print messages indicating need for user intervention."""
    global problem_found  # noqa: PLW0603
    problem_found = True
    if color_output:
        text = "\n".join(f"\033[31m{line}\033[0m" for line in text.splitlines())
    print(text, file=stderr)


def print_yellow(text: str) -> None:
    """Print messages indicating changes were made that the user should confirm."""
    global changes_made  # noqa: PLW0603
    changes_made = True
    if quiet_mode:
        return
    if color_output:
        text = "\n".join(f"\033[33m{line}\033[0m" for line in text.splitlines())
    print(text, file=stderr)


def print_green(text: str) -> None:
    """Print messages indicating no changes needed."""
    if quiet_mode:
        return
    if color_output:
        text = "\n".join(f"\033[32m{line}\033[0m" for line in text.splitlines())
    print(text, file=stderr)


def get_mailmap_path() -> Path:
    """Use git to to find the relative path to the .mailmap file."""
    cmd = ["git", "rev-parse", "--show-cdup"]
    res = subprocess.run(cmd, capture_output=True, check=True, text=True)
    return Path(res.stdout.strip()) / ".mailmap"


def get_raw_authors() -> dict[str, set[tuple]]:
    """Get the list of authors WITHOUT .mailmap remapping.

    Returns a dict {lowercase_email: set((name, email))}, which will be used to check if a
    mailmap line is necessary (matches some commit, and actually makes a change)
    """
    cmd = [
        "git",
        "log",
        "--pretty=tformat:%an <%ae>%n%(trailers:key=co-authored-by,valueonly=true,unfold=true)",
    ]
    res = subprocess.run(cmd, capture_output=True, check=True, text=True)

    logline = re.compile(r"([^<]+)\s+<(.*)>$")

    emails_map: dict[str, set[tuple]] = {}
    for line in res.stdout.splitlines():
        match = logline.fullmatch(line.strip())
        if match:
            name, email = match.group(1, 2)
            emails_map.setdefault(email.lower(), set()).add((name, email))
    return emails_map


def get_shortlog_authors() -> tuple[list[tuple], list[str]]:
    """Get the list of authors from git-shortlog (which uses .mailmap).

    Returns (matches: (num_commits, name, email), non_matches)
    """
    cmd = ["git", "shortlog", "-se", "--group=author", "--group=trailer:co-authored-by"]
    res = subprocess.run(cmd, capture_output=True, check=True, text=True)

    matches = []
    nonmatches = []
    shortlogline = re.compile(r"^(\d+)\s+([^<]+)\s+<(.*)>$")
    for line in res.stdout.splitlines():
        match = shortlogline.fullmatch(line.strip())
        if match:
            matches.append((int(match.group(1)), *match.group(2, 3)))
        else:
            # These are usually faulty co-author lines missing <> around the email address eg
            # Co-Authored-By: firstname lastname email@example.com
            nonmatches.append(line)
    return matches, nonmatches


def split_mailmap_header_footer(mailmap_file: Path) -> tuple:
    """Return the mailmap split into header comment lines and body lines."""
    mailmap = mailmap_file.read_text(encoding="UTF8").splitlines()
    for i, line in enumerate(mailmap):
        if not line.startswith("#"):
            header_end = i
            break
    header_lines = mailmap[:header_end]
    mailmap_lines = mailmap[header_end:]
    return header_lines, mailmap_lines


def parse_mailmap(mailmap_lines: list[str], raw_authors: dict):
    """Parse the mailmap body, making several checks and fixes.

    1. Warn on non-standard lines
    2. Remove name-matching
    3. Drop exact duplicate lines
    4. Warn for partial duplicate lines
    5. Drop lines that don't match a commit
    6. Drop lines that match a commit but cause no changes
    7. Alphabetise the body
    8. Warn about multiple canonical names associated with same email (we assume project-unique names)
    9. Warn about multiple canonical emails associated with same name (we assume project-unique emails)
    """
    line_re = re.compile(r"^([^<]+)\s+<([^>]*)>(?:(\s+[^<]+)?\s+<([^>]*)>)?$")
    output_lines = []  # Transformed lines with fixes applied
    email_mapping: dict[str, tuple] = {}  # Mapping from match-email to (canon-name, email, index)
    canonical_emails: dict[str, set[str]] = {}  # Mapping from canonical-email to {canonical-name}
    canonical_names: dict[str, set[str]] = {}  # Mapping from canonical-name to {canonical-email}

    for index, raw_line in enumerate(mailmap_lines):
        line, *comment = raw_line.split("#")
        line = line.strip()
        match = line_re.fullmatch(line)
        if not match:
            print_red(
                f"Found non-canonical mailmap line {mailmap_line(index)} "
                f"Please fix manually:\n    {raw_line}"
            )
            output_lines.append(raw_line)
            continue
        canonical_name, canonical_email, name2, email2 = match.group(1, 2, 3, 4)
        match_email = canonical_email if email2 is None else email2
        match_email = match_email.lower()  # git finds matches on lowercase of name, email

        # We could extend this tool to handle name matching if there's ever a case where it's needed.
        # But currently our history has no cases where multiple contributors share a commit email.
        if match_email.lower() not in raw_authors:
            print_yellow(
                f"Dropped the mailmap line {mailmap_line(index)} for <{match_email}> "
                "that doesn't match any author of a commit"
            )
            continue
        if match_email == canonical_email and raw_authors[match_email] == {
            (canonical_name, canonical_email)
        }:
            print_yellow(
                f"Dropped the mailmap line {mailmap_line(index)} for <{match_email}> "
                "that matches but doesn't cause any change",
            )
            continue

        if name2 is not None:
            print_yellow(
                f"Dropped mailmap name-match {mailmap_line(index)}:\n    {raw_line}",
            )
            line = f"{canonical_name} <{canonical_email}> <{email2}>"
        # if email2 == canonical_email:
        #     print_yellow(
        #         f"Dropped mailmap superfluous second email <{email2}> that is equal to the first "
        #         f"{mailmap_line(index)}"
        #     )
        #     line = f"{canonical_name} <{canonical_email}>"

        entry = (canonical_name, canonical_email, index)
        old_entry = email_mapping.setdefault(match_email, entry)
        if old_entry is not entry:
            if old_entry[:2] == entry[:2]:
                print_yellow(
                    f"Dropping exact duplicate mailmap line for <{match_email}> "
                    f"{mailmap_line(entry[2])} (previous was at {mailmap_line(old_entry[2])})"
                )
                continue
            print_red(
                f"Found inconsistent mailmap duplicates for <{match_email}>. "
                "Leaving both lines for manual fix:\n"
                f"    {mailmap_line(old_entry[2])}\n"
                f"        {old_entry[0]} <{old_entry[1]}>\n"
                f"    {mailmap_line(entry[2])}\n"
                f"        {entry[0]} <{entry[1]}>",
            )
        canonical_names.setdefault(canonical_name, set()).add(canonical_email)
        canonical_emails.setdefault(canonical_email, set()).add(canonical_name)

        output_lines.append(" #".join((line, *comment)))
    output_lines_sorted = sorted(output_lines, key=str.lower)
    if output_lines_sorted != output_lines:
        print_yellow("Changed the mailmap to alphabetic order.")
    else:
        print_green("Mailmap was already alphabetized.")

    for name, emails in canonical_names.items():
        if len(emails) == 1:
            continue
        print_red(f"Multiple canonical emails associated with canonical name '{name}':")
        for email in emails:
            print_red(f"    <{email}> from:")
            for match_email, (cname, cemail, index) in email_mapping.items():
                if cname == name and cemail == email:
                    print_red(f"        Matching <{match_email}> at {mailmap_line(index)}")
    for email, names in canonical_emails.items():
        if len(names) == 1:
            continue
        print_red(f"Multiple canonical names associated with canonical email <{email}>:")
        for name in names:
            print_red(f"    '{name}' from:")
            for match_email, (cname, cemail, index) in email_mapping.items():
                if cname == name and cemail == email:
                    print_red(f"        Matching <{match_email}> at {mailmap_line(index)}")

    return output_lines_sorted, email_mapping


match_non_alpha = re.compile(r"[^a-z]+")


def normalize_email(email: str) -> str:
    """Heuristically simplify email to spot accidental duplication."""
    return match_non_alpha.sub("", email.lower())


def normalize_name(name: str) -> str:
    """Heuristically simplify name to spot accidental duplication."""
    unidecoded = normalize("NFKD", name).encode("ascii", "ignore").decode()
    return match_non_alpha.sub("", unidecoded.lower())


def check_duplicates(authors: list, email_mapping: dict):
    """Find duplicated names, emails and try to suggest mailmap lines to combine them."""
    names: dict[str, set] = {}
    emails: dict[str, set] = {}
    for author in authors:
        _count, name, email = author
        names.setdefault(normalize_name(name), set()).add(author)
        emails.setdefault(normalize_email(email), set()).add(author)
    duplicated_names = {name: others for name, others in names.items() if len(others) > 1}
    duplicated_emails = {email: names for email, names in emails.items() if len(names) > 1}

    canonical_names = set()
    canonical_emails = set()
    for entry in email_mapping.values():
        canonical_name, canonical_email, _index = entry
        canonical_names.add(canonical_name)
        canonical_emails.add(canonical_email)

    new_mailmap_lines = set()
    for norm_name, dup_authors in duplicated_names.items():
        msg = f"Found potentially-duplicated authors whose names normalize to '{norm_name}'\n"

        canonical_name, msg1 = find_canonical_name(dup_authors, canonical_names)
        msg += msg1
        if canonical_name is None:
            print_red(msg)
            continue

        canonical_email, msg2 = find_canonical_email(dup_authors, canonical_emails)
        msg += msg2
        if canonical_email is None:
            print_red(msg)
            continue

        print_yellow(msg)
        for _count, match_name, match_email in dup_authors:
            if match_name == canonical_name and match_email == canonical_email:
                continue
            if match_email == canonical_email:
                new_mailmap_lines.add(f"{canonical_name} <{canonical_email}>")
            else:
                new_mailmap_lines.add(
                    f"{canonical_name} <{canonical_email}> <{match_email.lower()}>"
                )
    if new_mailmap_lines:
        return sorted(new_mailmap_lines, key=str.lower)
    for norm_email, dup_authors in duplicated_emails.items():
        msg = f"Found potentially-duplicated authors whose emails normalize to <{norm_email}>'\n"

        canonical_name, msg1 = find_canonical_name(dup_authors, canonical_names)
        msg += msg1
        if canonical_name is None:
            print_red(msg)
            continue

        canonical_email, msg2 = find_canonical_email(dup_authors, canonical_emails)
        msg += msg2
        if canonical_email is None:
            print_red(msg)
            continue

        print_yellow(msg)
        for _count, match_name, match_email in dup_authors:
            if match_name == canonical_name and match_email == canonical_email:
                continue
            if match_email == canonical_email:
                new_mailmap_lines.add(f"{canonical_name} <{canonical_email}>")
            else:
                new_mailmap_lines.add(
                    f"{canonical_name} <{canonical_email}> <{match_email.lower()}>"
                )
    return sorted(new_mailmap_lines, key=str.lower)


def find_canonical_name(dup_authors: set, canonical_names: set):
    """Search for a canonical name for a set of authors."""
    dup_names = tuple({d[1] for d in dup_authors})
    dup_names_canonical = tuple(d in canonical_names for d in dup_names)
    canonical_name = None
    if len(dup_names) == 1:
        canonical_name = dup_names[0]
        msg = f"    Only one unique name '{canonical_name}'\n"
    elif all(dup_names_canonical):
        names_list = ", ".join(f"'{n}'" for n in dup_names)
        msg = f"    All {len(names_list)} names are in mailmap: {names_list}\n"
        msg += "    This tool doesn't handle when normalization merges two unique authors."
        # If this comes up, either split the list of duplicates,
        # or do less-aggressive normalization
        return msg, None
    elif dup_names_canonical.count(True) == 1:
        canonical_name = dup_names[dup_names_canonical.index(True)]
        names_list = ", ".join(f"'{n}'" for n in dup_names if n != canonical_name)
        msg = f"    One name is in the mailmap: '{canonical_name}'\n"
        msg += f"        Setting the others ({names_list}) to match it\n"
    elif dup_names_canonical.count(True) == 0:
        counted_names = [(d, sum(a[0] for a in dup_authors if a[1] == d)) for d in dup_names]
        counted_names.sort(key=lambda x: x[1], reverse=True)
        canonical_name = counted_names[0][0]
        msg = f"    No names are in mailmap. Choosing '{canonical_name}' as the most common.\n"
        namecount = (f"'{c[0]}'({c[1]})" for c in counted_names)
        namecountjoin = ", ".join(namecount)
        msg += f"        {namecountjoin}\n"
    else:
        msg = "     No obvious choice for canonical name. Please resolve manually."
    return canonical_name, msg


def find_canonical_email(dup_authors: set, canonical_emails: set):
    """Search for a canonical email for a set of authors.

    Prefer:
        Unique input email
        Unique email mentioned in mailmap
        Unique github-noreply email
        Unique @ibm.com email
        Most common email from git log
    """
    dup_emails = tuple({d[2] for d in dup_authors})
    dup_emails_canonical = tuple(d in canonical_emails for d in dup_emails)
    canonical_email = None
    if len(dup_emails) == 1:
        canonical_email = dup_emails[0]
        msg = f"    One unique email <{canonical_email}>\n"
    elif all(dup_emails_canonical):
        emails_list = ", ".join(f"<{e}>" for e in dup_emails)
        msg = f"    All {len(emails_list)} names are in mailmap: {emails_list}\n"
        msg += "    This tool doesn't handle when normalization merges two unique authors."
        # If this comes up, either split the list of duplicates,
        # or do less-aggressive normalization
    elif dup_emails_canonical.count(True) == 1:
        canonical_email = dup_emails[dup_emails_canonical.index(True)]
        emails_list = ", ".join(f"<{n}>" for n in dup_emails if n != canonical_email)
        msg = f"    One email is in the mailmap: <{canonical_email}>\n"
        msg += f"        Setting the others ({emails_list}) to match it\n"
    elif dup_emails_canonical.count(True) == 0:
        dup_emails_github = tuple(d.endswith("noreply.github.com") for d in dup_emails)
        dup_emails_ibm = tuple(d.endswith("ibm.com") for d in dup_emails)
        counted_emails = [(d, sum(a[0] for a in dup_authors if a[2] == d)) for d in dup_emails]
        counted_emails.sort(key=lambda x: x[1], reverse=True)
        emailcount_list = ", ".join(f"<{c[0]}>({c[1]})" for c in counted_emails)

        if dup_emails_github.count(True) == 1:
            canonical_email = dup_emails[dup_emails_github.index(True)]
            emails_list = ", ".join(f"<{e}>" for e in dup_emails if e != canonical_email)
            msg = f"    One email is of github-noreply form: <{canonical_email}>\n"
            msg += f"    Setting the others ({emails_list}) to match it\n"
        elif dup_emails_github.count(True) > 1:
            msg = "   Multiple emails of github-noreply form. "
            msg += f"Please edit mailmap manually: {emailcount_list}"
        elif dup_emails_ibm.count(True) == 1:
            canonical_email = dup_emails[dup_emails_ibm.index(True)]
            emails_list = ", ".join(f"<{e}>" for e in dup_emails if e != canonical_email)
            msg = f"    One email is of @ibm.com form: <{canonical_email}>\n"
            msg += f"    Setting the others ({emails_list}) to match it\n"
        elif dup_emails_ibm.count(True) > 1:
            msg = f"   Multiple emails of @ibm.com form. Please edit mailmap manually: {emailcount_list}"
        else:
            canonical_email = counted_emails[0][0]
            msg = (
                f"    No emails are in mailmap. Choosing <{canonical_email}> as the most common.\n"
            )
            msg += f"        {emailcount_list}\n"
    else:
        msg = "     No obvious choice for canonical email. Please resolve manually."
    return canonical_email, msg


def _main():
    exit_code_string = "\n    ".join(f"{x.value}:\t{x.name}" for x in ExitCode)
    parser = argparse.ArgumentParser(
        description="Check .mailmap file and repo history for mailmapping problems",
        epilog="Exit codes:\n    " + exit_code_string,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    git_mailmap_path = get_mailmap_path()
    parser.add_argument(
        "--mailmap-file",
        default=git_mailmap_path,
        type=Path,
        help=f"Override mailmap file from {git_mailmap_path}",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quiet mode: suppress non-warning messages",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colored output",
    )
    parser.add_argument(
        "-s",
        "--skip-non-canonical-author",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Warn on non canonical authors (usually resulting from badly-formatted Co-Authored-By
        lines). There's usually no way to fix these short of rewriting git history""",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Don't actually change the mailmap, even when this tool considers it safe to do so""",
    )

    args = parser.parse_args()
    global mailmap_path, mailmap_body_offset, quiet_mode, color_output  # noqa: PLW0603
    mailmap_path = args.mailmap_file
    quiet_mode = args.quiet
    color_output = args.color

    # Load the raw author entries
    raw_authors = get_raw_authors()
    if len(raw_authors) < 10:
        print_red("Problem: `get_raw_authors()` returned too few authors")
        sys.exit(ExitCode.GIT_LOG)

    # Check the mailmap for internal consistency and necessity of entries
    if not mailmap_path.is_file():
        print_red(f"Didn't find a .mailmap file at '{mailmap_path}'")
        sys.exit(ExitCode.MAILMAP_NOT_FOUND)
    header_lines, mailmap_body_lines = split_mailmap_header_footer(args.mailmap_file)
    if len(mailmap_body_lines) < 10:
        print_red("Problem: too few mailmap body lines")
        sys.exit(ExitCode.MAILMAP_FORMAT)
    mailmap_body_offset = len(header_lines) + 1
    new_lines, email_mapping = parse_mailmap(mailmap_body_lines, raw_authors)
    if problem_found:
        print_red(
            "Problems were found with mailmap requiring manual fix. "
            "Skipping check for duplicates. Please re-run this tool after fixing them."
        )
        sys.exit(ExitCode.MAILMAP_NEEDS_MANUAL_CHANGE)
    if changes_made:
        if args.dry_run:
            print_yellow(
                "Changes were not to mailmap due to dry-run setting. "
                "Skipping check for duplicates. Please re-run this tool after making changes."
            )
            print("\n".join(header_lines + new_lines))
            sys.exit(ExitCode.MAILMAP_NEEDS_MANUAL_CHANGE)

        with open(mailmap_path, "w", encoding="utf8") as fptr:
            for line in itertools.chain(header_lines, new_lines):
                print(line, file=fptr)
        print_yellow(
            "Changes were made to mailmap. "
            "Skipping check for duplicates. Please re-run this tool after confirming them."
        )
        sys.exit(ExitCode.CHANGES_MADE)

    # Check the author entries with mailmap applied
    matches, nonmatches = get_shortlog_authors()
    if len(matches) < 10:
        print_red("Problem: `get_shortlog_authors()` returned too few authors")
        sys.exit(ExitCode.GIT_LOG)
    if not args.skip_non_canonical_author:
        msg = "Non-canonical committer lines:\n    " + "\n    ".join(nonmatches)
        print_green(msg)

    # Check for duplicated authors
    new_mailmap_lines = check_duplicates(matches, email_mapping)
    new_body_lines = sorted(set(mailmap_body_lines) | set(new_mailmap_lines), key=str.lower)

    if new_body_lines != mailmap_body_lines:
        if args.dry_run:
            print_yellow("Changes were not to mailmap due to dry-run setting.")
            print("\n".join(header_lines + new_body_lines))
            sys.exit(ExitCode.MAILMAP_NEEDS_MANUAL_CHANGE)
        else:
            with open(mailmap_path, "w", encoding="utf8") as fptr:
                for line in itertools.chain(header_lines, new_body_lines):
                    print(line, file=fptr)
            print_yellow(
                "Changes were made to mailmap. Please re-run this tool after confirming them."
            )
        sys.exit(ExitCode.CHANGES_MADE)

    if problem_found:
        print_red(
            "Not all duplicates were able to be automatically resolved. "
            "Please re-run this tool after manually resolving them."
        )
        sys.exit(ExitCode.MAILMAP_NEEDS_MANUAL_CHANGE)

    print_green("No problems were found in the mailmap")
    sys.exit(ExitCode.NO_PROBLEM)


if __name__ == "__main__":
    _main()
