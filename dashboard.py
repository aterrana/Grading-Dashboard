# Code author: Gabriel Nilsson - gabriel.nilsson@uni.minerva.edu

from plotly.express.colors import sample_colorscale
from plotly.express.colors import qualitative as disc_colors
import plotly.graph_objects as go
from scipy import stats as sts
import random as rnd
import numpy as np
import copy
import math

import os
import pathlib

import pandas as pd

class GradingDashboard:
    '''
    Class to manage navigation and visualisation of a given grade_data file.
    A grade_data file contains information about a single assignment across all current sections
    of that course.

    1. Create an instance of the class, 
    2. Run all methods corresponding to the graphs and components you want in the report (in the order you want them to appear)
    3. Run create_html to create the .html file of the report
    '''

    # Stored this list to avoid dependency on matplotlib
    forum_scoreclrs_rgbscheme = [
        [1.        , 0.        , 0.        , 1.        ],
        [1.        , 0.02029988, 0.        , 1.        ],
        [1.        , 0.05074971, 0.        , 1.        ],
        [1.        , 0.0710496 , 0.        , 1.        ],
        [1.        , 0.10149942, 0.        , 1.        ],
        [1.        , 0.12179931, 0.        , 1.        ],
        [1.        , 0.15224913, 0.        , 1.        ],
        [1.        , 0.18269896, 0.        , 1.        ],
        [1.        , 0.20299885, 0.        , 1.        ],
        [1.        , 0.23344867, 0.        , 1.        ],
        [1.        , 0.25374856, 0.        , 1.        ],
        [1.        , 0.28419839, 0.        , 1.        ],
        [1.        , 0.31464821, 0.        , 1.        ],
        [1.        , 0.3349481 , 0.        , 1.        ],
        [1.        , 0.36539792, 0.        , 1.        ],
        [1.        , 0.38569781, 0.        , 1.        ],
        [1.        , 0.41614764, 0.        , 1.        ],
        [1.        , 0.43644752, 0.        , 1.        ],
        [1.        , 0.46689735, 0.        , 1.        ],
        [1.        , 0.49734717, 0.        , 1.        ],
        [1.        , 0.51764706, 0.        , 1.        ],
        [1.        , 0.54809689, 0.        , 1.        ],
        [1.        , 0.56839677, 0.        , 1.        ],
        [1.        , 0.5988466 , 0.        , 1.        ],
        [1.        , 0.62929642, 0.        , 1.        ],
        [0.99607843, 0.64648981, 0.        , 1.        ],
        [0.94901961, 0.63966167, 0.        , 1.        ],
        [0.91764706, 0.63510957, 0.        , 1.        ],
        [0.87058824, 0.62828143, 0.        , 1.        ],
        [0.83921569, 0.62372933, 0.        , 1.        ],
        [0.79215686, 0.61690119, 0.        , 1.        ],
        [0.74509804, 0.61007305, 0.        , 1.        ],
        [0.71372549, 0.60552095, 0.        , 1.        ],
        [0.66666667, 0.59869281, 0.        , 1.        ],
        [0.63529412, 0.59414072, 0.        , 1.        ],
        [0.58823529, 0.58731257, 0.        , 1.        ],
        [0.54117647, 0.58048443, 0.        , 1.        ],
        [0.50980392, 0.57593233, 0.        , 1.        ],
        [0.4627451 , 0.56910419, 0.        , 1.        ],
        [0.43137255, 0.5645521 , 0.        , 1.        ],
        [0.38431373, 0.55772395, 0.        , 1.        ],
        [0.3372549 , 0.55089581, 0.        , 1.        ],
        [0.30588235, 0.54634371, 0.        , 1.        ],
        [0.25882353, 0.53951557, 0.        , 1.        ],
        [0.22745098, 0.53496348, 0.        , 1.        ],
        [0.18039216, 0.52813533, 0.        , 1.        ],
        [0.14901961, 0.52358324, 0.        , 1.        ],
        [0.10196078, 0.51675509, 0.        , 1.        ],
        [0.05490196, 0.50992695, 0.        , 1.        ],
        [0.02352941, 0.50537486, 0.        , 1.        ],
        [0.        , 0.49014994, 0.02352941, 1.        ],
        [0.        , 0.47440215, 0.05490196, 1.        ],
        [0.        , 0.45078047, 0.10196078, 1.        ],
        [0.        , 0.42715879, 0.14901961, 1.        ],
        [0.        , 0.411411  , 0.18039216, 1.        ],
        [0.        , 0.38778931, 0.22745098, 1.        ],
        [0.        , 0.37204152, 0.25882353, 1.        ],
        [0.        , 0.34841984, 0.30588235, 1.        ],
        [0.        , 0.33267205, 0.3372549 , 1.        ],
        [0.        , 0.30905037, 0.38431373, 1.        ],
        [0.        , 0.28542868, 0.43137255, 1.        ],
        [0.        , 0.26968089, 0.4627451 , 1.        ],
        [0.        , 0.24605921, 0.50980392, 1.        ],
        [0.        , 0.23031142, 0.54117647, 1.        ],
        [0.        , 0.20668973, 0.58823529, 1.        ],
        [0.        , 0.18306805, 0.63529412, 1.        ],
        [0.        , 0.16732026, 0.66666667, 1.        ],
        [0.        , 0.14369858, 0.71372549, 1.        ],
        [0.        , 0.12795079, 0.74509804, 1.        ],
        [0.        , 0.1043291 , 0.79215686, 1.        ],
        [0.        , 0.08070742, 0.83921569, 1.        ],
        [0.        , 0.06495963, 0.87058824, 1.        ],
        [0.        , 0.04133795, 0.91764706, 1.        ],
        [0.        , 0.02559016, 0.94901961, 1.        ],
        [0.        , 0.00196847, 0.99607843, 1.        ],
        [0.01377932, 0.        , 0.98632834, 1.        ],
        [0.037401  , 0.        , 0.9628912 , 1.        ],
        [0.06102268, 0.        , 0.93945406, 1.        ],
        [0.07677047, 0.        , 0.9238293 , 1.        ],
        [0.10039216, 0.        , 0.90039216, 1.        ],
        [0.11613995, 0.        , 0.8847674 , 1.        ],
        [0.13976163, 0.        , 0.86133026, 1.        ],
        [0.16338331, 0.        , 0.83789312, 1.        ],
        [0.1791311 , 0.        , 0.82226836, 1.        ],
        [0.20275279, 0.        , 0.79883122, 1.        ],
        [0.21850058, 0.        , 0.78320646, 1.        ],
        [0.24212226, 0.        , 0.75976932, 1.        ],
        [0.25787005, 0.        , 0.74414456, 1.        ],
        [0.28149173, 0.        , 0.72070742, 1.        ],
        [0.30511342, 0.        , 0.69727028, 1.        ],
        [0.32086121, 0.        , 0.68164552, 1.        ],
        [0.34448289, 0.        , 0.65820838, 1.        ],
        [0.36023068, 0.        , 0.64258362, 1.        ],
        [0.38385236, 0.        , 0.61914648, 1.        ],
        [0.40747405, 0.        , 0.59570934, 1.        ],
        [0.42322184, 0.        , 0.58008458, 1.        ],
        [0.44684352, 0.        , 0.55664744, 1.        ],
        [0.46259131, 0.        , 0.54102268, 1.        ],
        [0.486213  , 0.        , 0.51758554, 1.        ],
        [0.50196078, 0.        , 0.50196078, 1.        ]]

    def __init__(self, file_name: str, anonymize: bool = False, target_scorecount: int = None, student_count: list = []) -> None:
        '''
        Sets up global lists and dictionaries
        '''
        self.anonymize = anonymize

        # Read in data

        # Open the file
        with open(file_name, 'r', encoding='utf-8') as file:
            data = file.read()

        # Dangerous eval call, should also test for errors
        try:
            self.dict_all = eval(data)
        except:
            raise Exception("text in file is not a properly formatted dictionary")
        else:
            # Make it robust to errors here
            ...

        grades_dict = self.dict_all["grades"]

        self.target_scorecount = target_scorecount

        # Keep track of section ids, average scores, LO names, globally
        self.section_ids = list(grades_dict.keys())
        
        self.student_count = [len(grades_dict[section_id].keys()) for section_id in self.section_ids]
        
        if anonymize:
            # Shuffle key_order to anonymize report
            key_order = list(grades_dict.keys())
            rnd.shuffle(key_order)
            self.grades_dict = {k : grades_dict[k] for k in key_order}

            self.section_names = [f'Section {chr(i+65)}' for i in range(len(self.section_ids))]
        else:
            if len(self.section_ids) < 6:
                self.section_names = [self.dict_all['sections'][section_id]['title'].replace(' ', '<br>') for section_id in self.section_ids]
            else:
                self.section_names = [self.dict_all['sections'][section_id]['title'] for section_id in self.section_ids]
            self.grades_dict = grades_dict

        # Need to re-set section_ids so that they're in the same order as the potentially shuffled key order
        self.section_ids = list(self.grades_dict.keys())

        self.all_scores = []
        self.all_LOs = set()

        self.section_ids_w_scores = copy.deepcopy(self.section_ids)
        for section_id in self.section_ids:
            # A sublist for this section
            self.all_scores.append([])
            for student_id in self.grades_dict[section_id]:
                # A sublist for this student
                self.all_scores[-1].append([])
                for submission_data in self.grades_dict[section_id][student_id]:
                    if submission_data['score'] is not None:
                        # Add score for this specific student
                        self.all_scores[-1][-1].append(submission_data['score'])
                    if submission_data['learning_outcome'] != None:
                        self.all_LOs.add(submission_data['learning_outcome'])
                # If this student had no scores, omit the list
                if len(self.all_scores[-1][-1]) == 0:
                    self.all_scores[-1] = self.all_scores[-1][:-1]

            # If this section had no scores yet, occurs when only comments are exist
            if len(self.all_scores[-1]) == 0:
                print("NO SCORES")
                self.all_scores[-1].append([np.nan])
                # Keep note of the section ids that actually have scores
                self.section_ids_w_scores.remove(section_id)

        self.all_avgscores = []
        for i, section_id in enumerate(self.section_ids):
            #avg_scores = self.section_avg_scores(section_id_w_scores)
            # add a sublist for this section
            self.all_avgscores.append([])
            if section_id in self.section_ids_w_scores:
                for student_scores in self.all_scores[i]:
                    self.all_avgscores[-1].append(np.mean(student_scores))
            else:
                self.all_avgscores[-1].append(np.nan)
        
        # Set up the colors for the different sections
        self.section_colors = disc_colors.Dark24[:len(self.section_ids)]

        # set up section colors for the tables
        # Convert hexadecimal representation to rgb
        table_section_colors = [tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4)) for color in self.section_colors]
        # Increase brightness with 20%
        table_section_colors = [(min(255, int(rgb_val[0]*1.2)), min(255, int(rgb_val[1]*1.2)), min(255, int(rgb_val[2]*1.2))) for rgb_val in table_section_colors]
        # Convert to string and set alpha=0.6
        self.table_section_colors = [f'rgba({r}, {g}, {b}, 0.6)' for r, g, b in table_section_colors]

        self.sorted_LOs = self.get_sorted_LOs()

        # The list of all figures (or html text), in the right order, to be included in the report html
        self.figures = []
    
    def progress_table(self) -> None:
        ''' Produces table with progress indication, adds it to report '''

        # Count number of scores and comments
        scores_counts = []
        comments_counts = []
        comment_wordcounts = []
        for i, section_id in enumerate(self.section_ids):
            score_counter = 0
            comment_counter = 0
            comment_wordcounter = 0
            student_count = 0
            for student_id in self.grades_dict[section_id].keys():
                student_count += 1
                for submission_data in self.grades_dict[section_id][student_id]:
                    if submission_data['score'] is not None:
                        score_counter += 1
                    if len(submission_data['comment']) > 1:
                        comment_counter += 1
                        comment_wordcounter += len(submission_data['comment'].split(' '))
            if len(self.student_count) == 0:
                scores_counts.append(round(score_counter/student_count,2))
            else:
                scores_counts.append(round(score_counter/self.student_count[i],2))
            comments_counts.append(round(comment_counter/student_count,2))
            comment_wordcounts.append(round(comment_wordcounter/student_count,2))

        # Setup colors for score count
        if self.target_scorecount is None:
            scores_colorscale = ['lightblue']
            score_color_indices = [0] * len(scores_counts)
        else:
            #num_colors = 5
            #scores_colorscale = sample_colorscale('RdYlGn', list(np.linspace(0.15, 0.85, num_colors)))
            #score_color_indices = [min(num_colors-1, int((num_colors-1)*score_count/self.target_scorecount)) for score_count in scores_counts]

            scores_colorscale = ['rgb(255, 71, 76)', 'rgb(144, 238, 144)']
            score_color_indices = [0 if score_count < self.target_scorecount else 1 for score_count in scores_counts]



        redblue_colorscale = sample_colorscale('Greys', list(np.linspace(0, 1, 101)))

        # If the max count equals the min count, display the average color for all
        if max(comments_counts) == min(comments_counts):
            comm_count_color_indices = [10] * len(comments_counts)
        else:
            # Otherwise display the linear change
            comm_count_color_indices = [int(10+40*(val-min(comments_counts))/(max(comments_counts)-min(comments_counts))) for val in comments_counts]
        if max(comment_wordcounts) == min(comment_wordcounts):
            word_count_color_indices = [10] * len(comment_wordcounts)
        else:
            word_count_color_indices = [int(10+40*(val-min(comment_wordcounts))/(max(comment_wordcounts)-min(comment_wordcounts))) for val in comment_wordcounts]

        # Collect all columns in one list
        data = [self.section_names,
                scores_counts,
                comments_counts,
                comment_wordcounts]
        
        # Create table figure, with appropriate colors
        fig = go.Figure(data=[go.Table(header=dict(values=
                                                    ['Section name', 
                                                     'Scores per student',
                                                     'Comments per student',
                                                     'Comment wordcount per student']),
                                        cells=dict(values=data,
                                                    fill_color=[
                                                        self.table_section_colors,
                                                        np.array(scores_colorscale)[score_color_indices],
                                                        np.array(redblue_colorscale)[comm_count_color_indices],
                                                        np.array(redblue_colorscale)[word_count_color_indices]
                                                    ],
                                                    height=30))])
        
        # Make a dataframe of the data, it's more easily sortable
        df = pd.DataFrame(data).T.sort_values(0).T

        # Create dictionaries for all colors, with section_id as key, to be able to maintain colors after sorting
        section_color_dict = {section_id:self.table_section_colors[i] for i, section_id in enumerate(self.section_names)}
        scores_color_dict = {section_id:scores_colorscale[score_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        comm_color_dict = {section_id:redblue_colorscale[comm_count_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        word_color_dict = {section_id:redblue_colorscale[word_count_color_indices[i]] for i, section_id in enumerate(self.section_names)}


        # Create sorting drop-down menu
        fig.update_layout(
            updatemenus=[dict(
                    buttons= [dict(
                            method= "restyle",
                            label= selection["name"],
                                                                                            # Sort ascending only for column 0
                            args= [{"cells": {"values": df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values, # Sort all values according to selection
                                                "fill": dict(color=[[section_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]], # Ensure all colors are with the correct cell
                                                        [scores_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]],
                                                        [comm_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]],
                                                        [word_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]]
                                                        ]),
                                                "height": 30}},[0]]
                            )
                            for selection in [{"name": "Sort by section name", "col_i": 0}, 
                                      {"name": "Sort by score count", "col_i": 1}, 
                                      {"name": "Sort by comment count", "col_i": 2}, 
                                      {"name": "Sort by wordcount", "col_i": 3}]
                    ],
                    direction = "down",
                    y = 1.3,
                    x = 1
                )],
                height=270+29*len(self.section_ids) * (1 if self.anonymize else 2.1),
                font_size=15)
        
        self.figures.append('<center><h2>Grading progress summary</h2></center>')
        self.figures.append("This table summarizes the grading progress for each section.<br>")
        self.figures.append(f'The "scores per student" column will be colored more and more green as the value approaches the target number of scores per student, which for this assignment has been set to {self.target_scorecount}.<br>')
        self.figures.append("<br>Try out sorting each column using the dropdown menu to the right.<br>")
        if self.anonymize:
            self.figures.append("<i>The key associating each section ('Section B') with its Forum title ('Lastname MW@12:00pm, City') is found at the bottom of the report</i>")

        #self.figures.append('''<center><details><summary>Progress table</summary><p>''')
        self.figures.append(fig)
        #self.figures.append('''  </p> </details></center>''')

    def LO_progress_table(self) -> None:
        ''' Produces table with LO score-count progress, adds it to report '''

        # Count number of scores for each LO
        lo_scores_counts = [[0 for _ in range(len(self.section_ids))] for _ in range(len(self.sorted_LOs))]
        for lo_index, lo_name in enumerate(self.sorted_LOs):
            for section_index, section_id in enumerate(self.section_ids):
                for student_id in self.grades_dict[section_id].keys():
                    for submission_data in self.grades_dict[section_id][student_id]:
                        if submission_data['learning_outcome'] == lo_name:
                            lo_scores_counts[lo_index][section_index] += 1


        # Calculate total number of scores per section
        total_scores_given = np.array(lo_scores_counts[0])
        for lo_index in range(1, len(lo_scores_counts)):
            total_scores_given += np.array(lo_scores_counts[lo_index])

        greys_colorscale = sample_colorscale('Greys', list(np.linspace(0, 1, 101)))

        # Figure out what color to assign each cell for each LO, in order of LO appearance in self.all_LOs
        lo_color_indices = [[] for _ in range(len(self.sorted_LOs))]
        for lo_index in range(len(self.all_LOs)):

            # If the max count equals the min count, display the average color for all cells
            if max(lo_scores_counts[lo_index]) == min(lo_scores_counts[lo_index]):
                lo_color_indices[lo_index] = [25] * len(self.section_ids)
            else:
                # Otherwise display the linear change
                lo_color_indices[lo_index] = [int(50*(val-min(lo_scores_counts[lo_index]))/(max(lo_scores_counts[lo_index])-min(lo_scores_counts[lo_index]))) for val in lo_scores_counts[lo_index]]

        tot_lo_color_indices = []
        if max(total_scores_given) == min(total_scores_given):
            tot_lo_color_indices = [10] * len(self.section_ids)
        else:
            # Otherwise display the linear change
            tot_lo_color_indices = [int(10+40*(val-min(total_scores_given))/(max(total_scores_given)-min(total_scores_given))) for val in total_scores_given]
        # Collect all columns in one list
        data = [self.section_names, total_scores_given]
        data.extend(lo_scores_counts)

        column_names = ['Section name', 'Total scores assigned']
        for lo_name in self.sorted_LOs:
            column_names.append('Score count in ' + lo_name)
        
        column_colors = [self.table_section_colors, np.array(greys_colorscale)[tot_lo_color_indices]]
        
        column_colors.extend([np.array(greys_colorscale)[lo_color_indices[i]] for i in range(len(self.sorted_LOs))])
        
        self.figures.append('<center><h2>LO grading progress</h2></center>')
        self.figures.append('This table shows how many scores each section has, in total and for each LO')
        
        # Split up the data, colors, and names into multiple lists, that go into multiple tables
        data_copy = data[:]
        column_colors_copy = column_colors[:]
        column_names_copy = column_names[:]

        # Figure out how many tables will be needed, and what their width should be
        max_width = 8
        num_tables = math.ceil(len(data_copy) / max_width)
        table_width = math.ceil((len(data_copy) + num_tables-1) / num_tables)

        if len(data_copy) >= table_width:
            
            split_data = []
            split_column_colors = []
            split_column_names = []
            
            while len(data_copy) >= table_width:
                split_data.append(data_copy[:table_width])
                split_column_colors.append(column_colors_copy[:table_width])
                split_column_names.append(column_names_copy[:table_width])

                data_copy = data_copy[table_width:]
                data_copy.insert(0, data[0])
                column_colors_copy = column_colors_copy[table_width:]
                column_colors_copy.insert(0, column_colors[0])
                column_names_copy = column_names_copy[table_width:]
                column_names_copy.insert(0, column_names[0])
            
            if len(data_copy) > 1:
                split_data.append(data_copy[:])
                split_column_colors.append(column_colors_copy[:])
                split_column_names.append(column_names_copy[:])


        else:
            split_data = [data]
            split_column_colors = [column_colors]
            split_column_names = [column_names]
        
        for j, data in enumerate(split_data):
            # Create table figure, with appropriate colors
            fig = go.Figure(data=[go.Table(header=dict(values=split_column_names[j]),
                                            cells=dict(values=data,
                                                        fill_color=split_column_colors[j],
                                                        height=30))])
            
            # Make a dataframe of the data, it's more easily sortable
            df = pd.DataFrame(data).T.sort_values(0).T

            # Create dictionaries for all colors, with section_id as key, to be able to maintain colors after sorting
            section_color_dict = {section_id:self.table_section_colors[i] for i, section_id in enumerate(self.section_names)}

            total_color_dict = {section_id:split_column_colors[j][1][i] for i, section_id in enumerate(self.section_names)}

            lo_color_dicts = [{section_id:split_column_colors[j][lo_name_i+1][i] for i, section_id in enumerate(self.section_names)} for lo_name_i in range(len(data)-1)]

            ## Create sorting drop-down menu
            fig.update_layout(
                updatemenus=[dict(
                        buttons= [dict(
                                method= "restyle",
                                label= selection["name"],
                                args= [{"cells": {"values": df.T.sort_values(selection["col_i"], ascending=selection['col_i']==0).T.values, # Sort all values according to selection
                                                    "fill": dict(color=[
                                                            [section_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]],
                                                            [total_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]], # Ensure all colors are with the correct cell
                                                            *[[color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]] for color_dict in lo_color_dicts]]),
                                                    "height": 30}}, [0]]
                                )
                                for selection in [
                                    {"name": "Sort by section name", "col_i": 0},
                                    {"name": f"Sort by {'total scores' if j==0 else split_column_names[j][1].split(' ')[-1]}", "col_i": 1},
                                    *[{"name": f"Sort by {lo_name}", "col_i": i+2} for i, lo_name in enumerate(split_column_names[j][2:])]
                                ]
                        ],
                        direction = "down",
                        y = 1.26,
                        x = 1
                    )],
                    height=125+29*len(self.section_ids) * (1 if self.anonymize else 2.8),
                    font_size=15,
                    margin = dict(t=0, b=0))
            
            self.figures.append(fig)
            
    def summary_stats_table(self) -> None:
        ''' Produces table with summary statistics, adds it to report '''
        # Calculate section means and SDs
        self.section_means = np.array([np.mean(section_avgs) for section_avgs in self.all_avgscores])
        self.section_SDs = np.array([np.std(section_avgs) if len(section_avgs) > 0 else np.nan for section_avgs in self.all_avgscores])

        # Number of students in each section
        self.section_sizes = np.array([len(self.grades_dict[section_id].keys()) for section_id in self.section_ids])

        # Overall average score and SDs, weighted by number of students. This defines the "average" section
        overall_mean = sum([val for val in (self.section_means * self.section_sizes) if not np.isnan(val)])/sum(self.section_sizes)

        overall_SD = sum([val for val in (self.section_SDs * self.section_sizes) if not np.isnan(val)])/sum(self.section_sizes)
        #average_section_size = np.mean(self.section_sizes)

        signif_count = np.zeros(len(self.section_names), int)
        cohens_d_vals = np.zeros(len(self.section_names), float)

        # For each group pair
        for a_i in range(len(self.section_names)):
            for b_i in range(a_i+1, len(self.section_names)):
                # Count the number of other sections this section is significantly different with
                _, p_value = sts.ttest_ind(self.all_avgscores[a_i], self.all_avgscores[b_i], nan_policy='omit')

                # If this pair is significantly different, increment corresponding position in signif_count
                if p_value < 0.05:
                    signif_count[a_i] += 1
                    signif_count[b_i] += 1

                    # Calculate cohen's d between the two groups
                    # Calculate effect size (Cohen's d), https://en.wikipedia.org/wiki/Effect_size#:~:text=large.%5B23%5D-,Cohen%27s,-d%20%5Bedit
                    # variance of each group
                    var_a = np.var(self.all_avgscores[a_i], ddof=1)
                    var_b = np.var(self.all_avgscores[b_i], ddof=1)

                    # Pooled standard deviation
                    pooled_sd = np.sqrt(((len(self.all_avgscores[a_i])-1)*var_a + (len(self.all_avgscores[b_i])-1)*var_b ) / (len(self.all_avgscores[a_i] + self.all_avgscores[b_i])-2) )
                    cohens_d = (np.mean(self.all_avgscores[a_i]) - np.mean(self.all_avgscores[b_i])) / pooled_sd
                    
                    # Store the accumulate absolute effect size, will be divided by count later
                    cohens_d_vals[a_i] += abs(cohens_d)
                    cohens_d_vals[b_i] += abs(cohens_d)

        for i in range(len(self.section_names)):
            if signif_count[i] == 0:
                cohens_d_vals[i] = np.nan
            else:
                cohens_d_vals[i] = round(cohens_d_vals[i] / signif_count[i], 2)

        # Assign colors to cells

        # Create colorscales
        redblue_colorscale = self.forum_scoreclrs_rgbscheme # Grab this from const variable in class

        bright_f = 1.3 # Increased brightness of score colors by 30%
        redblue_colorscale = ["rgba(" + str(min(255, int(255*val[0]*bright_f))) + ',' + str(min(255, int(255*val[1]*bright_f))) + ',' + str(min(255, int(255*val[2]*bright_f))) + ",0.6)" for val in redblue_colorscale]

        greys_colorscale = sample_colorscale('Greys', list(np.linspace(0, 1, 101)))
        reds_colorscale = sample_colorscale('Reds', list(np.linspace(0, 1, 101)))

        # If all means are equal
        if max(self.section_means)-min(self.section_means) == 0 or np.isnan(max(self.section_means)) or np.isnan(min(self.section_means)):
            # Set all colors to the middle point
            mean_color_indices = [50] * len(self.section_means)
        else:
            # Let means vary linearly from 0.15 to 0.85 on redblue colorscale
            mean_color_indices = []
            for mean_val in self.section_means:
                if np.isnan(mean_val):
                    mean_color_indices.append(50)
                else:
                    mean_color_indices.append(int(25*(mean_val-1)))
        # If all SDs are equal
        if max(self.section_SDs)-min(self.section_SDs) == 0 or np.isnan(max(self.section_SDs)) or np.isnan(min(self.section_SDs)):
            # Set all colors to the middle point
            sd_color_indices = [10] * len(self.section_SDs)
        else:
            # Let SDs vary linearly from 0.0 to 0.7 on yellowred colorscale
            sd_color_indices = []
            for SD_val in self.section_SDs:
                if np.isnan(SD_val):
                    sd_color_indices.append(0)
                else:
                    sd_color_indices.append(int(10+40*(SD_val-min(self.section_SDs))/(max(self.section_SDs)-min(self.section_SDs))))
        
        # precalc the min and max according to absolute value of effect size
        cohens_d_numbers = [0 if np.isnan(val) else val for val in cohens_d_vals]
        
        min_abs_effect = min([abs(val) for val in cohens_d_numbers])
        max_abs_effect = max([abs(val) for val in cohens_d_numbers])

        # If all effect sizes are equal
        if max_abs_effect - min_abs_effect == 0 or math.isinf(max_abs_effect):
            effect_color_indices = [0] * len(cohens_d_vals)
        else:
            # Let effect size vary linearly from 0.0 to 0.7 on yellowred colorscale
            effect_color_indices = [int(10+60*(abs(val)-min_abs_effect)/(max_abs_effect-min_abs_effect)) if val != 0 else 0 for val in cohens_d_numbers]

        # precalc the min and max according to absolute value of effect size
        min_signif_count = min([val for val in signif_count]) 
        max_signif_count = max([val for val in signif_count])

        # If all effect sizes are equal
        if max_signif_count - min_signif_count == 0:
            signif_count_color_indices = [0] * len(signif_count)
        else:
            # Convert nan to 0's for color indices
            signif_count_color_indices = [int(60*(abs(val)-min_signif_count)/(max_signif_count-min_signif_count)) for val in signif_count]
        
                
        # Create table
        data = [self.section_names[:],
                [round(val,3) for val in self.section_means],
                [round(val,3) for val in self.section_SDs],
                [round(val,5) for val in signif_count],
                [round(val,5) for val in cohens_d_vals]]
        
        data[0].insert(0, '<b>Average</b>')
        this_table_section_colors = self.table_section_colors[:]
        this_table_section_colors.insert(0, 'white')

        data[1].insert(0, '<b>' + str(round(overall_mean, 3)) + '</b>')
        mean_colors = list(np.array(redblue_colorscale)[mean_color_indices])
        mean_colors.insert(0, 'white')

        data[2].insert(0, '<b>' + str(round(overall_SD, 3)) + '</b>')
        sd_color_indices.insert(0, 0)
        data[3].insert(0, '<b>NA</b>')
        signif_count_color_indices.insert(0, 0)
        data[4].insert(0, '<b>NA</b>')
        effect_color_indices.insert(0, 0)

        for column_i in range(1, 5):
            for i in range(len(data[column_i])):
                data[column_i][i] = str(data[column_i][i])
                if data[column_i][i] == 'nan':
                    if column_i != 3:
                        data[column_i][i] = ' NA'
                    else:
                        data[column_i][i] = 'NA'


        fig = go.Figure(data=[go.Table(header=dict(values=
                                                    ['Section name', 
                                                     'Mean score',
                                                     'Standard Deviation',
                                                     'Count of significant test results',
                                                     'Avg effect size among significant']),
                                        cells=dict(values=data,
                                                    fill_color=[
                                                        this_table_section_colors,
                                                        mean_colors,
                                                        np.array(greys_colorscale)[sd_color_indices],
                                                        np.array(reds_colorscale)[signif_count_color_indices],
                                                        np.array(reds_colorscale)[effect_color_indices]
                                                    ],
                                                    height=30))])
        
        # Make a dataframe of the data, it's more easily sortable
        df = pd.DataFrame(data).T.sort_values(0).T

        # Create dictionaries for all colors, with section_id as key, to be able to maintain colors after sorting
        section_color_dict = {section_id:this_table_section_colors[i] for i, section_id in enumerate(data[0])}
        mean_color_dict = {section_id:mean_colors[i] for i, section_id in enumerate(data[0])}
        sd_color_dict = {section_id:greys_colorscale[sd_color_indices[i]] for i, section_id in enumerate(data[0])}
        signf_count_dict = {section_id:reds_colorscale[signif_count_color_indices[i]] for i, section_id in enumerate(data[0])}
        effect_color_dict = {section_id:reds_colorscale[effect_color_indices[i]] for i, section_id in enumerate(data[0])}
        

        # Create sorting drop-down menu
        fig.update_layout(
            updatemenus=[dict(
                    buttons= [dict(
                            method= "restyle",
                            label= selection["name"],
                                                                                            # Sort ascending only for column 0 and 3
                            args= [{"cells": {"values": df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values, # Sort all values according to selected column
                                                "fill": dict(color=[[section_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]], # Ensure colors are with correct cell
                                                        [mean_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]],
                                                        [sd_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]],
                                                        [signf_count_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]],
                                                        [effect_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0).T.values[0]]
                                                        ]), 
                                                "height": 30}},[0]]
                            )
                            for selection in [{"name": "Sort by section name", "col_i": 0}, 
                                      {"name": "Sort by mean", "col_i": 1}, 
                                      {"name": "Sort by standard deviation", "col_i": 2}, 
                                      {"name": "Sort by signif count", "col_i": 3}, 
                                      {"name": "Sort by avg effect size", "col_i": 4}]
                    ],
                    direction = "down",
                    y = 1.2,
                    x = 1
                )],
                height=290+30*len(self.section_ids) * (1 if self.anonymize else 3),
                font_size=15)

        self.figures.append('This table summarizes the information from pairwise t-tests between each section. The detailed information of each t-test can be found below this table.<br>')
        self.figures.append('The "Count of significant test results" describes how many other sections this specific section is different from, with statistical significance (p<0.05).<br>')
        #self.figures.append('If there is a "problem section", this would show up as one section having a clearly larger number in this column.<br>')
        self.figures.append("The final column \"Avg effect size among significant\" displays the average Cohen's d among all tests with a statistically significant result. Cohen’s d is a standardized measure of practical significance, calculated as the difference in means in units of standard deviations:.<br>")

        self.figures.append(fig)

    def t_test_grids(self) -> None:
        '''
        Normal difference of means test, assumes normality of data 
        Assummptions:
        - continuous variables
        - Normal distribution of data
        - independent groups
        - Sufficient sample size (preferable more than ~20)
        '''

        # Calculate p-value between each group
        cohens_d_vals = np.zeros((len(self.section_names), len(self.section_names)))
        cohens_d_text = [['' for _ in range(len(self.section_names))] for _ in range(len(self.section_names))]
        t_test_pvals_text = [['' for _ in range(len(self.section_names))] for _ in range(len(self.section_names))]
        t_test_pvals_colori = np.zeros((len(self.section_names), len(self.section_names)))
        for a_i in range(len(self.section_names)):
            for b_i in range(len(self.section_names)):
                
                # Populate upper triangle with nan
                if b_i < a_i:
                    cohens_d_text[b_i][a_i] = ''
                    cohens_d_vals[a_i, b_i] = np.nan
                    t_test_pvals_text[b_i][a_i] = ''
                    t_test_pvals_colori[a_i, b_i] = np.nan
                    continue
                elif b_i == a_i:
                    cohens_d_text[b_i][a_i] = '<b>X</b>'
                    cohens_d_vals[a_i, b_i] = 0
                    t_test_pvals_text[b_i][a_i] = '<b>X</b>'
                    t_test_pvals_colori[a_i, b_i] = 0.6
                    continue
                else:
                    t_test_pvals_colori[a_i, b_i] = 0.73

                if len(self.all_avgscores[a_i]) > 0 and len(self.all_avgscores[b_i]) > 0:
                    if not np.isnan(self.all_avgscores[a_i][0]) and not np.isnan(self.all_avgscores[b_i][0]):

                        # Calculate p_value
                        _, p_value = sts.ttest_ind(self.all_avgscores[a_i], self.all_avgscores[b_i], nan_policy='omit')

                        # Calculate effect size (Cohen's d), https://en.wikipedia.org/wiki/Effect_size#:~:text=large.%5B23%5D-,Cohen%27s,-d%20%5Bedit
                        # variance of each group
                        var_a = np.var(self.all_avgscores[a_i], ddof=1)
                        var_b = np.var(self.all_avgscores[b_i], ddof=1)

                        # Pooled standard deviation
                        pooled_sd = np.sqrt( ( (len(self.all_avgscores[a_i])-1)*var_a + (len(self.all_avgscores[b_i])-1)*var_b ) / (len(self.all_avgscores[a_i] + self.all_avgscores[b_i])-2) )
                        cohens_d = (np.mean(self.all_avgscores[a_i]) - np.mean(self.all_avgscores[b_i])) / pooled_sd

                        cohens_d_vals[a_i, b_i] = round(cohens_d, 2)
                        cohens_d_text[b_i][a_i] = str(round(cohens_d, 2))
                        t_test_pvals_text[b_i][a_i] = str(round(p_value, 5))
                        if p_value < 0.05 and not p_value == 0:
                            t_test_pvals_colori[a_i, b_i] = 0.1
                        elif p_value < 0.1:
                            t_test_pvals_colori[a_i, b_i] = 0.25

        # Do some flipping around so that it becomes a lower triangle down-left
        t_test_pvals_text = [list(row) for row in reversed(t_test_pvals_text)]
        cohens_d_text = [list(row) for row in reversed(cohens_d_text)]
        t_test_pvals_colori = np.transpose(np.array(t_test_pvals_colori))
        t_test_pvals_colori = [list(row) for row in reversed(t_test_pvals_colori)]

        fig = go.Figure(data=go.Heatmap(z=t_test_pvals_colori,
                                        zmax=1,
                                        zmin=0,
                                        x=self.section_names,
                                        y=list(reversed(self.section_names)), 
                                        customdata=t_test_pvals_text,
                                        text=t_test_pvals_text,
                                        name='',
                                        hovertemplate='%{x}<br>%{y}<br>%{customdata:.5f}',
                                        texttemplate="%{text}",
                                        hoverongaps=False,
                                        colorscale='RdBu'))
        
        fig.update_layout(plot_bgcolor='white',
                          title="<b>p-values from T tests</b><br>Significant tests (p<.05) is colored in red")
        fig.update_traces(showscale=False)
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)

        self.figures.append(fig)

        # Coordinates to highlight
        highlight_coords = []
        for a_i in range(len(self.section_names)-1):
            for b_i in range(len(self.section_names)-a_i-1):
                if float(t_test_pvals_text[a_i][b_i]) < 0.05:
                    highlight_coords.append((self.section_names[b_i], self.section_names[len(self.section_names)-a_i-1]))
        highlight_coords = [val for val in highlight_coords for _ in range(2)]

        # Scatter trace for highlighting
        highlight_scatter = go.Scatter(
            x=[x for x, _ in highlight_coords],
            y=[y for _, y in highlight_coords],
            mode='markers',
            marker=dict(
                size=14,  # Adjust the size as needed
                symbol='triangle-up',
                line=dict(
                    color='black',
                    width=1.5
                ),
                color='rgba(1,1,1,1)',  # Fully transparent fill
                angle=[270, 90] * len(highlight_coords),
                standoff=26 # 20 for arrow-wide
            ),
            showlegend=False,
            hoverinfo='skip'
        )

        # Do some flipping around so that it becomes a lower triangle down-left
        cohens_d_vals = np.transpose(np.array(cohens_d_vals))
        cohens_d_vals = [list(row) for row in reversed(cohens_d_vals)]
        fig = go.Heatmap(z=cohens_d_vals,
                                        zmax=1,
                                        zmin=-1,
                                        x=self.section_names,
                                        y=list(reversed(self.section_names)), 
                                        customdata=cohens_d_vals,
                                        text=cohens_d_text,
                                        name='',
                                        hovertemplate='%{x}<br>%{y}<br>%{customdata:.2f}',
                                        texttemplate="%{text}",
                                        hoverongaps=False,
                                        colorscale='RdBu')
        
        fig = go.Figure(data=[fig, highlight_scatter])
        
        fig.update_layout(plot_bgcolor='white',
                          title="<b>Effect sizes (cohen's d) from T tests</b><br>The number of standard deviations away the means are.<br>Positive values mean that x-axis section has larger mean than y-axis section.")
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)

        self.figures.append(fig)

    # Not used currently
    def mann_whitney_grid(self) -> None:
        '''
        Mann-Whitney tests whether two groups are different or not (come from the same distribution)
        Variables are assumed to be continuous (so student averages, not individual scores),
        and data is assumed to be non-Normal (skewed), 
        Assummptions:
        - continuous variables
        - non-Normal data, but similar in shape between groups
        - independent groups
        - Sufficient sample size (usually more than 5)
        '''

        # Calculate mann-whitney test between each combination of groups.
        # Calculate the Mann-Whitney U test
        # null hypothesis is that the prob of X > Y equals prob of Y > X
        mannwhitney_ustats = np.zeros((len(self.section_names), len(self.section_names)))
        mannwhitney_pvals = np.zeros((len(self.section_names), len(self.section_names)))
        mannwhitney_pvals_signif = np.zeros((len(self.section_names), len(self.section_names)))
        mannwhitney_pvals_signif_text = [['' for _ in range(len(self.section_names))] for _ in range(len(self.section_names))]
        wilcoxon_data = np.zeros((len(self.section_names), len(self.section_names)))
        for a_i in range(len(self.section_names)):
            for b_i in range(len(self.section_names)):
                
                # Populate upper triangle with nan
                if b_i < a_i:
                    mannwhitney_ustats[a_i, b_i] = np.nan
                    mannwhitney_pvals[a_i, b_i] = np.nan
                    mannwhitney_pvals_signif[a_i, b_i] = np.nan
                    continue
                elif b_i == a_i:
                    mannwhitney_ustats[a_i, b_i] = 0
                    mannwhitney_pvals[a_i, b_i] = 0
                    mannwhitney_pvals_signif[a_i, b_i] = 0.6
                    mannwhitney_pvals_signif_text[len(self.section_names)-b_i-1][a_i] = '<b>X</b>'

                    continue
                else:
                    mannwhitney_pvals_signif[a_i, b_i] = 0.73


                if len(self.all_avgscores[a_i]) > 0 and len(self.all_avgscores[b_i]) > 0:
                    if not np.isnan(self.all_avgscores[a_i][0]) and not np.isnan(self.all_avgscores[b_i][0]):
                        print(self.all_avgscores[a_i])
                        print(self.all_avgscores[b_i])
                        u_stat, p_value_mw = sts.mannwhitneyu(self.all_avgscores[a_i], self.all_avgscores[b_i], nan_policy='omit')

                        #stat, p_value_w = sts.wilcoxon(self.all_avgscores[a_i], self.all_avgscores[b_i], nan_policy='omit')
                        mannwhitney_ustats[a_i, b_i] = u_stat
                        mannwhitney_pvals[a_i, b_i] = p_value_mw
                        if a_i == 3 or b_i == 3:
                            print(p_value_mw)
                            print(p_value_mw == 0)
                        if p_value_mw < 0.05 and not p_value_mw == 0:
                            mannwhitney_pvals_signif[a_i, b_i] = 0.1
                            mannwhitney_pvals_signif_text[len(self.section_names)-b_i-1][a_i] = '<0.05'
                        elif p_value_mw < 0.1:
                            mannwhitney_pvals_signif[a_i, b_i] = 0.25
                            mannwhitney_pvals_signif_text[len(self.section_names)-b_i-1][a_i] = '<0.1'

                        #wilcoxon_data[a_i, b_i] = stat

        print(mannwhitney_ustats)
        print(mannwhitney_pvals)
        # Do some flipping around so that it becomes a lower triangle down-left
        mannwhitney_ustats = np.transpose(np.array(mannwhitney_ustats))
        mannwhitney_ustats = [list(row) for row in reversed(mannwhitney_ustats)]
        fig = go.Figure(data=go.Heatmap(z=mannwhitney_ustats,
                                        x=self.section_names,
                                        y=list(reversed(self.section_names)), 
                                        hoverongaps=False))
        
        fig.update_layout(plot_bgcolor='white',
                          title="U-statistic from Mann-Whitney tests")
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)

        self.figures.append(fig)

        #p_val_significance = np.array(mannwhitney_pvals) < 0.05
        # Do some flipping around so that it becomes a lower triangle down-left
        mannwhitney_pvals = np.transpose(np.array(mannwhitney_pvals))
        mannwhitney_pvals = [list(row) for row in reversed(mannwhitney_pvals)]
        fig = go.Figure(data=go.Heatmap(z=mannwhitney_pvals,
                                        x=self.section_names,
                                        y=list(reversed(self.section_names)), 
                                        hoverongaps=False))
        
        fig.update_layout(plot_bgcolor='white',
                          title="p-values from Mann-Whitney tests")
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)

        self.figures.append(fig)

        print(mannwhitney_pvals_signif)
        # Do some flipping around so that it becomes a lower triangle down-left
        mannwhitney_pvals_signif = np.transpose(np.array(mannwhitney_pvals_signif))
        mannwhitney_pvals_signif = [list(row) for row in reversed(mannwhitney_pvals_signif)]
        print(mannwhitney_pvals_signif)
        fig = go.Figure(data=go.Heatmap(z=mannwhitney_pvals_signif,
                                        zmax=1,
                                        zmin=0,
                                        x=self.section_names,
                                        y=list(reversed(self.section_names)),
                                        text=mannwhitney_pvals_signif_text,
                                        hoverongaps=False,
                                        customdata=mannwhitney_pvals,
                                        name='',
                                        hovertemplate='%{x}<br>%{y}<br>%{customdata:.5f}',
                                        texttemplate="%{text}",
                                        colorscale='RdBu'))
        
        fig.update_layout(plot_bgcolor='white', 
                          title="<b>Significance of Mann-Whitney test between each section</b><br>with a significance level of a=0.05")
        fig.update_traces(showscale=False)
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)

        self.figures.append(fig)

    # Not used currently
    def prob_of_superiority_grid(self) -> None:
        '''
        Mann-Whitney tests whether two groups are different or not (come from the same distribution)
        Variables are assumed to be continuous (so student averages, not individual scores),
        and data is assumed to be non-Normal (skewed), 
        Assummptions:
        - continuous variables
        - non-Normal data, but similar in shape between groups
        - independent groups
        - Sufficient sample size (usually more than 5)
        '''

        # Calculate mann-whitney test between each combination of groups.
        # Calculate the Mann-Whitney U test
        # null hypothesis is that the prob of X > Y equals prob of Y > X
        prob_superiority = np.zeros((len(self.section_names), len(self.section_names)))
        for a_i in range(len(self.section_names)):
            for b_i in range(len(self.section_names)):
                
                # Populate upper triangle with nan
                if b_i < a_i:
                    prob_superiority[a_i, b_i] = np.nan
                    continue
                elif b_i == a_i:
                    prob_superiority[a_i, b_i] = 0.5
                    continue


                if len(self.all_avgscores[a_i]) > 0 and len(self.all_avgscores[b_i]) > 0:
                    if not np.isnan(self.all_avgscores[a_i][0]) and not np.isnan(self.all_avgscores[b_i][0]):
                        print(self.all_avgscores[a_i])
                        print(self.all_avgscores[b_i])
                        
                        # Total number of pairings
                        tot_pairs = len(self.all_avgscores[a_i]) * len(self.all_avgscores[b_i])

                        # How many of those pairs is A superior
                        sup_count = 0
                        for a_val in self.all_avgscores[a_i]:
                            for b_val in self.all_avgscores[b_i]:
                                if a_val > b_val:
                                    sup_count += 1

                        prob_superiority[a_i, b_i] = sup_count / tot_pairs
                        


        print(prob_superiority)
        # Do some flipping around so that it becomes a lower triangle down-left
        prob_superiority = np.transpose(np.array(prob_superiority))
        prob_superiority = [list(row) for row in reversed(prob_superiority)]
        fig = go.Figure(data=go.Heatmap(z=prob_superiority,
                                        x=self.section_names,
                                        y=list(reversed(self.section_names)), 
                                        hoverongaps=False))
        
        fig.update_layout(plot_bgcolor='white',
                          title="Probability of superiority between each group")
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)
        
        self.figures.append(fig)
    
    def section_avg_scores(self, section_id: int) -> list:
        ''' Returns a list with the average score of each student for this section '''
        
        student_ids = list(self.grades_dict[section_id].keys())
        avg_scores = []

        # For each student
        for this_student_id in student_ids:
            # Calculate average score
            student_scores = []
            for score_data in self.grades_dict[section_id][this_student_id]:
                # Excluded None
                if score_data['score'] != None:
                    student_scores.append(score_data['score'])
            if len(student_scores) > 0:
                avg_scores.append(np.mean(student_scores))

        return avg_scores
    
    # Not used currently
    def section_scoreavgs_plot(self, section_id: int, bin_size:int=0.2) -> None:
        ''' Creates plot that shows the distribution of average scores, for given section'''

        # This didn't look good, need to change this plot

        section_avgs = self.section_avg_scores(section_id)

        fig = go.Figure(data=[go.Histogram(
                        x=section_avgs,
                        xbins=dict(
                            start=0,
                            end=5,
                            size=bin_size))])

        # Make edges white
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(
            title=f'Score distribution across section ({len(section_avgs)} students)',
            xaxis_title='Score',
            yaxis_title='Count')
        
        # Add plot to the report
        self.figures.append(fig)
    
    def scoreavgs_allsections_plot(self, bin_size:int=0.2) -> None:
        ''' Creates plot that shows a single histogram of student scores distribution '''

        # Flatten the list of scores
        all_scores_flat = [avg_score for section_lst in self.all_avgscores for avg_score in section_lst if not np.isnan(avg_score)]

        # Make histogram    
        fig = go.Figure(data=go.Histogram(
                    x=all_scores_flat,
                    xbins=dict(
                        start=0,
                        end=5,
                        size=bin_size),
                    opacity=0.8,
                    showlegend=False))
        
        # Manually calculate the height of the bins to write it as text
        counts = []
        left_edge = 0
        all_scores_flat_sorted = sorted(all_scores_flat)
        current_index = 0
        while left_edge < 5:
            left_edge += 0.2

            old_index = current_index
            # If this region contains scores
            while all_scores_flat_sorted[current_index] < left_edge + 0.1999:
                # count how many scores
                current_index += 1
                if current_index == len(all_scores_flat_sorted):
                    break
            # If there was any, add it to counts
            if current_index - old_index > 0:
                counts.append(current_index - old_index)
            # If there were none, but this is not first bin, add empty text
            elif len(counts) > 0:
                counts.append('')

            if current_index == len(all_scores_flat_sorted):
                break
                
        fig.add_vline(x=np.mean(all_scores_flat), annotation_text='Mean')
        fig.add_vline(x=np.median(all_scores_flat), annotation_text='Median', annotation_position="bottom right", fillcolor='orange')
        
        fig.update_traces(marker_line_width=1, marker_line_color="white", text=counts)
        # Count total number of students
        student_count_tot = len([student_id for section in self.grades_dict.keys() for student_id in self.grades_dict[section].keys() ]) 
        fig.update_layout(
            title=f'<b>Student score averages</b>, all sections combined<br>({len(self.section_ids)} sections, {student_count_tot} students)',
            xaxis_title='Score',
            yaxis_title='Count')
        
        fig.update_xaxes(dtick=0.2)
        
        # Add plot to the report
        self.figures.append(fig)

    def ANOVA_test(self, averages=True) -> None:
        ''' Performs oneway ANOVA test, and creates html text presenting the results. '''
        
        # Decides whether to consider each student as one average number
        # or each HC/LO score as one datapoint
        if averages:
            scores = self.all_avgscores
        else:
            scores = self.all_scores

        # Perform the ANOVA test
        try:
            # Remove all nan from scores
            filtered_scores = []
            for sublist in scores:
                if not np.isnan(sublist[0][0]):
                    filtered_scores.append([val for lst in sublist for val in lst])

            f_stat, p_value = sts.f_oneway(*filtered_scores)
            #f_stat, p_value = f_stat[0], p_value[0]
            if np.isnan(f_stat) or np.isnan(p_value):
                raise Exception("Not enough data")

            output = ''
            
            # Display the results
            # Small p-value means statistically significant
            output += "<center><h2> ANOVA Results (global significance test)</h2></center>"
            output += "ANOVA is a difference of means test for multiple groups. It gives a global p-value, which if significant, means that it's likely that there is a general association between student average score and section.<br>"
            output += "Note that ANOVA assumes normality, and equal variance in each group. Especially the second criteria could be false for student scores, and if so, ANOVA results should not be trusted blindly."
            rounded_pval_string = str(round(p_value, 4))
            pval_percentage_string = str(round(p_value*100, 2))
            if round(p_value, 4) == 0:
                rounded_pval_string = "<0.00005"
                pval_percentage_string = "<0.005"
            output += "<h3>P-value: " + rounded_pval_string + "</h3>"
            output += "(With significance level = 0.05)<br>"
            if p_value < 0.05:
                output += "We reject the null hypothesis. The different groups likely don't share the same true mean. There's a " + pval_percentage_string + "% chance of a Type I error.\n\n"
            else:
                output += "We don't reject the null hypothesis. We don't have statistically significant evidence that the groups have different true means.\n\n"
            # F-statistic: Variation between sample means / Variation within samples
            # Large F-statistic means that there is difference somewhere
            output += "<h3>F-statistic: " + str(round(f_stat, 3)) + "</h3>"
            output += "A larger F-statistic means more difference between groups.<br>"
            output += "An F-stat of " + str(round(f_stat, 3)) + " means that the variance between <b>sample means</b> is " + str(round(f_stat, 3)) + " times the <b>variance within samples</b>."

            # Add text to the report
            self.figures.append(output)
        except Exception as e:
            output = ''
            output += "<center><h1> ANOVA Results</h1></center>"
            output += "Couldn't perform ANOVA test because:<br>" + str(e)
            self.figures.append(output)

    def boxplots(self, averages=True) -> None:
        ''' Creates side by side boxplots, together with jittered scatterplots displaying student average scores '''

        # Whether considering student average scores, or all individual LO/HC scores
        # Deepcopy because we will edit local copy.
        if averages:
            scores = copy.deepcopy(self.all_avgscores)
        else:
            scores = copy.deepcopy(self.all_scores)

        fig = go.Figure()

        # Flatten scores list for plotting
        for section_scores in scores:
            # We need 5 datapoints for the mean and SD plotting to work
            # So if fewer, add None to make the difference
            if len(section_scores) < 5:
                diff = 5 - len(section_scores)
                section_scores.extend([None] * diff)

        # For a given section in this list,
        # All datapoints will be at the mean except
        # 2 datapoints which will be +1 and -1 SD.
        means_and_SDs = []

        # calcuate means and SDs
        for section_scores in scores:
            filtered_list = [score for score in section_scores if score is not None]
            # If there's any scores in this section
            this_mean_and_SD = []
            if len(filtered_list) > 0:
                mean = np.mean(filtered_list)
                std_dev = np.std(filtered_list)
                this_mean_and_SD.append(mean-std_dev)
                this_mean_and_SD.append(mean+std_dev)

                # Fill the rest of the section with the mean
                remaining_count = len(section_scores) - 2
                this_mean_and_SD.extend([mean] * remaining_count)
            # There are no scores in this section yet
            else:
                this_mean_and_SD.extend([None]*5)
            means_and_SDs.append(this_mean_and_SD)


        tickvals_list = []
        ticktext_list = []
        for i, group in enumerate(self.section_names):
            fig.add_trace(go.Box(
                y=scores[i],
                name=group,
                marker_color=self.section_colors[i],
                legendgroup=group,
                quartilemethod="inclusive",
                showlegend = True
            ))

            tickvals_list.append(group)
            ticktext_list.append(group)

            # Boxplot only displaying mean and SDs
            fig.add_trace(go.Box(
                y=means_and_SDs[i],
                name=group + '+',
                legendgroup=group,
                line_width=3,
                whiskerwidth=1,
                marker_color=self.section_colors[i],
                quartilemethod="inclusive",
                boxpoints=False,
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add a transparent scatterplot over the mean & SD boxplot to fix hover text
            fig.add_trace(go.Scatter(
                y=[max(means_and_SDs[i])],
                x=[group+'+'],
                legendgroup=group,
                marker_color='rgba(1,1,1,0)',
                showlegend=False,
                hovertemplate='Mean + 1sd: %{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                y=[means_and_SDs[i][-1]],
                x=[group+'+'],
                legendgroup=group,
                marker_color='rgba(1,1,1,0)',
                showlegend=False,
                hovertemplate='Mean: %{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                y=[min(means_and_SDs[i])],
                x=[group+'+'],
                legendgroup=group,
                marker_color='rgba(1,1,1,0)',
                showlegend=False,
                hovertemplate='Mean - 1sd: %{y:.2f}<extra></extra>'
            ))

            tickvals_list.append(group+'+')
            ticktext_list.append('')


        fig.update_layout(
            title='<b>Student score averages</b>, per section<br>Box plots are quartiles<br>adjacent whiskerplots are mean +/- 1sd',
            xaxis_title='Section',
            yaxis_title='Scores',
            height=800,
            legend=dict(groupclick="togglegroup"),
            xaxis=dict(
                tickvals=tickvals_list,
                ticktext=ticktext_list
            ),
            hovermode='x'
        )

        # Add plot to the report
        self.figures.append(fig)

    # Not used currently
    def violinplots(self, averages=True) -> None:
        ''' Creates side by side violinplots '''

        # Whether considering student average scores, or all individual LO/HC scores
        if averages:
            scores = self.all_avgscores
        else:
            scores = self.all_scores
        
        fig = go.Figure()
        for i, section_scores in enumerate(scores):
            fig.add_trace(go.Violin(x=section_scores, name=f'Section {chr(65+i)}'))

        fig.update_traces(orientation='h', side='positive', width=3, points=False, meanline_visible=True, meanline=dict(color='black', width=2))

        fig.update_layout(
            title='<b>Distribution of average score per section</b><br>kernel density estimate plot, black line is mean',
            xaxis_title='Scores',
            yaxis_title='Section',
            height=800)
        
        # Add plot to the report
        self.figures.append(fig)

    # Not used currently
    def LO_kde_plot(self, LO_name:str) -> None:
        ''' Creates side by side kde plots with mean, displaying general grade distribution per section '''
        
        # Find scores of this LO in each section
        LO_scores = []

        score_count = 0
        for section_id in self.section_ids:
            LO_scores.append([])
            for student_id in self.grades_dict[section_id]:
                for submission_data in self.grades_dict[section_id][student_id]:
                    if submission_data['learning_outcome'] == LO_name:
                        if submission_data['score'] is not None:
                            score_count += 1
                            LO_scores[-1].append(submission_data['score'])

        # Skip the plot if there's not enough scores in total
        if score_count < 4:
            print(f"Three or less {LO_name} scores were granted across all sections, and the kde LO plot was cancelled")
            return
        
        # To keep track of which sections we're actually displaying
        y_axes_labels = []

        # Create the kde plots
        fig = go.Figure()
        for i, section_scores in enumerate(LO_scores):
            fig.add_trace(go.Violin(x=section_scores, name=f'Section {chr(65+i)}', legendgroup=chr(65+i), points='all'))
            if len(section_scores) > 0:
                y_axes_labels.append(f'Section {chr(65+i)}')

        # Orient them horizontally, with a black mean line
        fig.update_traces(orientation='h', side='positive', width=3, points=False, meanline_visible=True, meanline=dict(color='black', width=2))

        fig.update_layout(
            title=f'<b>Distribution of average score in {LO_name} per section</b><br>kernel density estimate plot, black line is mean',
            xaxis_title='Scores',
            yaxis_title='Section',
            height=800)
        
        fig.update_yaxes(tickvals=y_axes_labels)
        
        # Add plot to the report
        self.figures.append(fig)

    def LO_stackedbar_plot(self, LO_name:str) -> None:
        ''' Create stacked barplot with percentages '''
        
        # Find scores of this LO in each section
        LO_scores_count = []

        for section_id in self.section_ids:
            # Add 5 counters for the 5 possible scores
            LO_scores_count.append(np.zeros(5))
            for student_id in self.grades_dict[section_id]:
                for submission_data in self.grades_dict[section_id][student_id]:
                    if submission_data['learning_outcome'] == LO_name:
                        # Increment the the counter corresponding to the score
                        try:
                            LO_scores_count[-1][int(submission_data['score'])-1] += 1
                        except:
                            ...
        
        LO_scores_perc = [np.zeros(5) for _ in range(len(self.section_ids))]
        # If there are any scores, convert counts to percentages [0-1]
        for i, section in enumerate(LO_scores_count):
            if sum(section) > 0:
                LO_scores_perc[i] = np.array(section)/sum(section)

        # The official minerva grade colors, from left to right, 1 to 5
        colors = ['rgba(223,47,38,255)', 'rgba(240,135,30,255)', 'rgba(51,171,111,255)', 'rgba(10,120,191,255)', 'rgba(91,62,151,255)']
        fig = go.Figure()
        for score in range(1, 6):
            score_fractions = [section_scores[score-1] for section_scores in LO_scores_perc]
            fig.add_trace(go.Bar(name=score, 
                                 x=self.section_names, 
                                 y=score_fractions, 
                                 marker=dict(color=colors[score-1]),
                                 text=[str(round(val*100, 1)) + '%<br>' + str(int(LO_scores_count[i][score-1])) for i, val in enumerate(score_fractions)],
                                 textposition='inside'))
            
        # Change the bar mode to stack
        fig.update_layout(
            title=f'<b>Score distribution for {LO_name} per section</b><br>Stacked barplots.',
            xaxis_title='Section',
            yaxis_title='Percentage',
            height=800,
            barmode='stack')
        
        fig.layout.yaxis.tickformat = '0%'

        # Add plot to the report
        self.figures.append(fig)
        
    def LO_stackedbar_plot_all(self) -> None:
        ''' Create stacked barplot with percentages '''
        
        # Find scores of this LO in each section
        LO_scores_count = []

        for LO_name in self.sorted_LOs:
            # Add 5 counters for the 5 possible scores
            LO_scores_count.append(np.zeros(5))
            # For each grade of all students
            for section_id in self.section_ids:
                for student_id in self.grades_dict[section_id]:
                    for submission_data in self.grades_dict[section_id][student_id]:
                        # If it matches this specific LO
                        if submission_data['learning_outcome'] == LO_name:
                            # Increment the the counter corresponding to the score
                            try:
                                LO_scores_count[-1][int(submission_data['score'])-1] += 1
                            except:
                                ...
            
        LO_scores_perc = [np.zeros(5) for _ in range(len(self.all_LOs))]
        # If there are any scores, convert counts to percentages [0-1]
        for i, this_lo_dist in enumerate(LO_scores_count):
            if sum(this_lo_dist) > 0:
                LO_scores_perc[i] = np.array(this_lo_dist)/sum(this_lo_dist)

        # The official minerva grade colors, from left to right, 1 to 5
        colors = ['rgba(223,47,38,255)', 'rgba(240,135,30,255)', 'rgba(51,171,111,255)', 'rgba(10,120,191,255)', 'rgba(91,62,151,255)']
        fig = go.Figure()
        for score in range(1, 6):
            score_fractions = [section_scores[score-1] for section_scores in LO_scores_perc]
            fig.add_trace(go.Bar(name=score,
                                 x=list(self.sorted_LOs), 
                                 y=score_fractions, 
                                 marker=dict(color=colors[score-1]),
                                 text=[str(round(val*100, 1)) + '%<br>' + str(int(LO_scores_count[i][score-1])) for i, val in enumerate(score_fractions)],
                                 textposition='inside'))
            
        # Change the bar mode to stack
        fig.update_layout(
            title=f'<b>Score distribution per LO</b><br>Stacked barplots.',
            xaxis_title='Learning Outcome',
            yaxis_title='Percentage',
            height=800,
            barmode='stack')
        
        for i, lo_name in enumerate(self.sorted_LOs):
            fig.add_annotation(x=lo_name, y=1, yshift=10,
                               text=f'{int(sum(LO_scores_count[i]))} Score' + ('s' if int(sum(LO_scores_count[i])) > 1 else ''), 
                               showarrow=False)
        
        fig.layout.yaxis.tickformat = '0%'

        # Add plot to the report
        self.figures.append(fig)

    def section_id_table(self) -> None:
        ''' A small table associated anonymous labels of sections to their true section title '''

        section_names = [self.dict_all['sections'][id]['title'] for id in self.section_ids]

        fig = go.Figure(data=[go.Table(header=dict(values=
                                                    ['Section Name in Report', 
                                                     'Section Title']),
                                        cells=dict(values=[self.section_names, section_names],
                                                    fill_color=[
                                                        self.table_section_colors
                                                    ],
                                                    height=30))])
        
        fig.update_layout(height=250+35*len(self.section_ids), font_size=15)

        self.figures.append('<center><h1>Section report name to section title key</h1></center>')
        
        self.figures.append(fig)

    def create_html(self) -> None:
        '''
        Add the appropriate css styling, and create an html containing the entire report 
        in the order of self.figures
        '''

        html_head = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <style>
                body {
                    font-family: 'Open Sans', sans-serif;
                }

                details {
                    user-select: none;
                }

                details>summary span.icon {
                    width: 24px;
                    height: 24px;
                    transition: all 0.3s;
                }

                details[open] summary span.icon {
                    transform: rotate(-90deg);
                }

                summary {
                    display: flex;
                    cursor: pointer;
                }

                summary::-webkit-details-marker {
                    display: none;
                }

                .vertical {
                    border-left: 2px solid black;
                }

                .spaced {
                margin-left: 40px;
                }
            </style>
        </head>
        <body>
        '''

        html_end = "</body>"

        self.figures.insert(0, html_head)
        self.figures.append(html_end)

        with open('grading_dashboard.html', 'a', encoding="utf-8") as f:
            # Remove contents
            f.truncate(0)
            for fig in self.figures:
                if isinstance(fig, str):
                    f.write(fig)
                else:
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        print('Dashboard complete')

    def get_sorted_LOs(self) -> list:
        ''' Returns a list of all graded LOs sorted in descending order according to the number of scores in that LO '''
        # Count the number of scores for each LO
        lo_score_counts = []
        for lo_name in self.all_LOs:
            score_count = 0
            for section_id in self.section_ids:
                for student_id in self.grades_dict[section_id]:
                    for submission_data in self.grades_dict[section_id][student_id]:
                        if submission_data['learning_outcome'] == lo_name:
                            score_count += 1
            lo_score_counts.append(score_count)

        # Sort the lo names according to the number of scores
        return [LO_name for _, LO_name in sorted(zip(lo_score_counts, self.all_LOs), key=lambda pair: pair[0], reverse=True)]

    def make_full_report(self) -> None:
        ''' Creates a pre-selected set of plots and results, in the right order, then creates html '''

        self.figures.append(f"<center><h1>Grading Dashboard for {self.dict_all['course']['code']}, {self.dict_all['assignment_title']}</h1></center>")
        self.figures.append("<center><h1>Grading Progress</h1></center>")
        self.figures.append('''<details><summary>Summary progress table  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.progress_table()
        except Exception as error_message: 
            print(f"Failed to create progress table\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        self.figures.append('''<br><details><summary>LO progress tables  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.LO_progress_table()
        except Exception as error_message: 
            print(f"Failed to create LO progress table\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        self.figures.append("<center><h1>Section Comparisons</h1></center>")
        self.figures.append("This section describes the comparisons between sections and their practical and statistical significance.<br>")
        self.figures.append("For all tests, independence and normality is assumed.<br>")
        self.figures.append('''<details><summary>ANOVA test results  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.ANOVA_test(False)
        except Exception as error_message: 
            print(f"Failed to create ANOVA test\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        self.figures.append('<center><h2>Summary statistics (Pairwise significance tests)</h2></center>')
        self.figures.append('''<details><summary>Summary stats table  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.summary_stats_table()
        except Exception as error_message: 
            print(f"Failed to create summary stats table\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')


        self.figures.append("<center><h1>Score distributions</h1></center>")
        self.figures.append('''<details><summary>Scores histogram  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.scoreavgs_allsections_plot()
        except Exception as error_message: 
            print(f"Failed to create score histogram\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        self.figures.append('''<br><details><summary>Scores boxplots  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        self.figures.append("In the figure below, each section has two plots.")
        self.figures.append("    The left one is a boxplot, showing the 4 quartiles of student scores. That means that the middle line is the median, having equally many student scores above and below it.")
        self.figures.append("    The right one is a whisker plot, showing the mean of the section, and one standard deviation above and below the mean.")
        self.figures.append("<b> Click or double click the legend on the right to select and deselect different sections</b>")
        try: self.boxplots()
        except Exception as error_message: 
            print(f"Failed to create score boxplots\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        self.figures.append("<center><h2>Pairwise significance test results (T tests)</h2></center>")
        self.figures.append('''<details><summary>T-test results  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.t_test_grids()
        except Exception as error_message: 
            print(f"Failed to create t-test result grid\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')
        
        self.figures.append("<center><h1>LO score distributions</h1></center>")
        self.figures.append('''<details><summary>Stacked barplot, per LOs  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        try: self.LO_stackedbar_plot_all()
        except Exception as error_message: 
            print(f"Failed to create stacked barplot for all LOs\n {error_message=}")
            self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        self.figures.append('''<br><details><summary>Stacked barplot, per section  <span class="icon">👈</span></summary><p>''')
        self.figures.append('<div class= "vertical"><div class= "spaced">')
        for lo_name in self.sorted_LOs:
            try: self.LO_stackedbar_plot(lo_name)
            except Exception as error_message: 
                print(f"Failed to create stacked barplot for {lo_name}\n {error_message=}")
                self.figures.append(f"This plot failed because: {error_message=}")
        self.figures.append('''  </div></div></p> </details>''')

        if self.anonymize:
            try: self.section_id_table()
            except Exception as error_message: print(f"Failed to create section id table\n {error_message=}")
        self.figures.append("<br><center><i>The report code and instructions can be found <a href='https://github.com/g-nilsson/Grading-Dashboard'>here</a>, written by <a href='mailto:gabriel.nilsson@uni.minerva.edu'>gabriel.nilsson@uni.minerva.edu</a>, reach out for questions</i></center>")
        self.figures.append("V2.0")
        self.create_html()

def create_report(anonymize, target_scorecount):
    print("Creating report")
    gd = GradingDashboard('grade_data.py', anonymize=anonymize, target_scorecount=target_scorecount)
    gd.make_full_report()

    print("Opening report")

    dir_path = pathlib.Path(__file__).parent.resolve()
    # Make sure this works on a mac as well, might need to do .as_posix() at least
    #os.system("open grading_dashboard.html")
    #os.system("open " f"{dir_path.as_posix()}/grading_dashboard.html")
    try:
        os.startfile(f'{dir_path}\grading_dashboard.html')
    except:
        try:
            file_path = f"{dir_path.as_posix()}/grading_dashboard.html"
            print(file_path)
            os.system(f'open "{file_path}"')
        except:
            print("Was unable to open the grading dashboard html, the html can be found at:")
            print(f'{dir_path}\grading_dashboard.html')

    # Clearing the grade_data file to not accidentally publish any grading data to the github repo
    with open("grade_data.py", 'w', encoding="utf-8") as file:
        file.write("")
