# Minerva Grading Dashboard

This project is an assignment grading dashboard for course leads, so that they can import the current grading data for a given assignment (across all active sections of the course) and observe differences in grading patterns between sections.

# How to set up

1. [Download](https://chromewebstore.google.com/detail/minerva-assignment-data-g/afmncekkkklpkommcpkgiielaoahiiin) the "Minerva Assignment Data-grabber" chrome extension.
2. [Download](https://github.com/aterrana/Grading-Dashboard/archive/refs/heads/main.zip) this project, remember the path to the folder after you've unzipped it.
3. If you haven't already, [install python](https://www.python.org/downloads/)
4. Install the python libraries: plotly, and requests-futures, by pasting the following commands in your command prompt `<br>`

```
pip install pandas
pip install plotly==5.23.0
pip install requests-futures
pip install scipy
```

If it says "pip: command not found", try typing "pip3" instead of "pip", try pasting this instead:

```
pip3 install pandas
pip3 install plotly==5.23.0
pip3 install requests-futures
pip3 install scipy
```

# How to use

1. (Optional) Pin the chrome extension`<br>`

<p align="center">
<img src="https://github.com/g-nilsson/public_files/blob/main/pin_extension.gif" width="280" />
</p>
2. Navigate to an assignment page in Forum (of the format forum.minerva.edu/app/assignments/xxxxxx)<br>
3. Press the button in the chrome extension popup. It'll paste the command you need into your clipboard<br>
<p align="center">
<img src="https://github.com/g-nilsson/public_files/blob/main/using_extension.gif" width="600" />
</p>
4. Open your command prompt (search cmd in Windows, and terminal on Mac)<br>
5. In your command prompt, navigate to the unzipped folder you downloaded in step 2 of "How to set up"<br>
<p align="center">
<img src="https://github.com/g-nilsson/public_files/blob/main/using_cd.gif" width="600" />
</p>
6. Paste the command that is already on your clipboard by pressing ctrl+V (Windows) or command+V<br>
7. Press enter<br>
<p align="center">
<img src="https://github.com/g-nilsson/public_files/blob/main/using_grading_dashboard.gif" width="600" />
</p>
<br>
The code should now run, and the Grading Dashboard should open, automatically. This might take a second.

# Functionality

The script will create an html file containing interactive graphs and statistical results for the given assignment across all sections.
The html will describe and summarize data like:

- ANOVA results, whether there's statistical evidence that at least one section is giving higher/lower scores
- Distribution of students' average assignment scores over all sections
- Distributions of students' LO scores over all sections
