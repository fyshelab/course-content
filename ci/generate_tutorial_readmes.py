"""Write a directory of tutorial notebooks to the README file.

Run this script from the root of the github repository.

"""
import os
from glob import glob


def main():

    # Initialize the lines in tutorials/README.md
    course_readme_text = [
    ]

    try:
        playlist_urls = load_youtube_playlist_urls()
    except Exception as err:
        print("Encountered error while loading youtube playlist links")
        print(err)
        playlist_urls = {}

    try:
        slide_urls = load_slide_urls()
    except Exception as err:
        print("Encountered error while loading slide links")
        print(err)
        slide_urls = {}

    day_anchors = {}

    day_paths = sorted(glob("tutorials/W?_*"))
    for day_path in day_paths:

        day_name = os.path.split(day_path)[-1]
        day_code, topic_code = day_name.split("_")

        # Split the UpperCamelCase topic name into separate words
        topic_words = []
        for letter in topic_code:
            if letter.isupper():
                topic_words.append(letter)
            else:
                topic_words[-1] += letter
        topic = " ".join(topic_words)

        # Note: this will fail if we have 10+ notebooks
        notebooks = sorted(glob(f"{day_path}/*.ipynb"))

        if not notebooks:
            continue

        # Track the anchor to this section for embed in the header
        anchor = "-".join([
            day_code.lower(),
            "-",
            ("-".join(topic_words)).lower(),
        ])
        if "W0" not in day_code:
            day_anchors[day_code] = "#" + anchor

        student_notebooks = get_student_links(notebooks)

        # Write the day information into the course README
        course_readme_text.extend([
            f"## {day_code} - {topic}",
            "",
        ])

        # Add a link to the YouTube lecture playlist, if we have one
        youtube_url = playlist_urls.get(day_code, None)
        if youtube_url is not None:
            course_readme_text.extend([
                f"[YouTube Playlist]({youtube_url})"
                "",
            ])

        slide_links_by_topic = slide_urls.get(day_code, None)
        if slide_links_by_topic is not None:
            slide_links = [
                f"[{topic}]({url})" for topic, url in slide_links_by_topic
            ]
            course_readme_text.extend([
                "",
                "Slides: " + " | ".join(slide_links),
                "",
            ])

        course_readme_text.extend(write_badge_table(student_notebooks))
        course_readme_text.append("\n")

        # Now make the day-specific README
        # with links to both instructor and student versions
        day_readme_text = [
            f"# {day_code} - {topic}",
            "",
            "## Instructor notebooks",
            "",
        ]
        day_readme_text.extend(write_badge_table(notebooks))

        day_readme_text.extend([
            "## Student notebooks",
            "",
        ])
        day_readme_text.extend(write_badge_table(student_notebooks))

        # Write the day README file
        with open(f"{day_path}/README.md", "w") as f:
            f.write("\n".join(day_readme_text))

    # Create relative anchor links to each day
    nav_line = " | ".join([
        f"[{day_code}]({anchor})" for day_code, anchor in day_anchors.items()
    ])

    # Add an introductory header to the main README
    course_readme_header = [
        "# CIS-522 Materials",
        "",
        "<!-- DO NOT EDIT THIS FILE. IT IS AUTO-GENERATED BY A FRIENDLY ROBOT -->",
        "",
        nav_line,
        "",
        "*Warning:* The 'render with NBViewer' buttons may show outdated content.",
        "",
    ]
    course_readme_text = course_readme_header + course_readme_text

    # Write the course README file
    with open("tutorials/README.md", "w") as f:
        f.write("\n".join(course_readme_text))


def load_youtube_playlist_urls():
    """Create a mapping from day code to youtube link based on text file."""
    with open("tutorials/youtube_playlists.txt") as f:
        lines = filter(bool, f.read().split("\n"))
    return dict(tuple(line.split()) for line in lines)


def load_slide_urls():
    """Create a hierarchical mapping to slide PDF urls based on text file."""
    with open("tutorials/slide_links.txt") as f:
        lines = filter(bool, f.read().split("\n"))
    slide_links = {}
    for line in lines:
        day, topic, url = line.split()
        if day not in slide_links:
            slide_links[day] = []
        slide_links[day].append((topic, url))
    return slide_links


def write_badge_table(notebooks):
    """Make a markdown table with colab/nbviewer badge links."""

    # Add the table header
    table_text = [
        "|   | Run | View |",
        "| - | --- | ---- |",
    ]

    # Add each row of the table
    for i, local_path in enumerate(notebooks, 1):

        colab_badge = make_colab_badge(local_path)
        nbviewer_badge = make_nbviewer_badge(local_path)
        table_text.append(
            f"| Tutorial {i} | {colab_badge} | {nbviewer_badge} |"
        )
    table_text.append("\n")

    return table_text


def get_student_links(instructor_notebooks):
    """Convert a list of instructor notebook paths to student versions."""
    student_notebooks = []
    for instructor_nb in instructor_notebooks:
        day_path, nb_fname = os.path.split(instructor_nb)
        student_notebooks.append(f"{day_path}/student/{nb_fname}")
    return student_notebooks


def make_colab_badge(local_path):
    """Generate a Google Colaboratory badge for a notebook on github."""
    alt_text = "Open In Colab"
    badge_svg = "https://colab.research.google.com/assets/colab-badge.svg"
    url_base = (
        "https://colab.research.google.com/"
        "github/CIS-522/course-content/blob/main"
    )
    return make_badge(alt_text, badge_svg, url_base, local_path)


def make_nbviewer_badge(local_path):
    """Generate an NBViewer badge for a notebook on github."""
    alt_text = "View the notebook"
    badge_svg = "https://img.shields.io/badge/render-nbviewer-orange.svg"
    url_base = (
        "https://nbviewer.jupyter.org/"
        "github/CIS-522/course-content/blob/main"
    )
    return make_badge(
        alt_text, badge_svg, url_base, f"{local_path}?flush_cache=true"
    )


def make_badge(alt_text, badge_svg, url_base, local_path):
    """Generate a markdown element for a badge image that links to a file."""
    return f"[![{alt_text}]({badge_svg})]({url_base}/{local_path})"


if __name__ == "__main__":

    main()
