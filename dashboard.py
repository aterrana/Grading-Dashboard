# Code author: Gabriel Nilsson - gabriel.nilsson@uni.minerva.edu

from plotly.express.colors import sample_colorscale
from plotly.express.colors import qualitative as disc_colors
import plotly.graph_objects as go
from scipy import stats as sts
import random as rnd
import numpy as np
import copy
import math

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
    def __init__(self, file_name: str, anonymize: bool = False, target_scorecount: int = None) -> None:
        '''
        Sets up global lists and dictionaries
        '''
        self.anonymize = anonymize

        # Read in data

        # Open the file
        with open(file_name, 'r', encoding='utf8') as file:
            data = file.read()

        # Dangerous eval call, should also test for errors
        try:
            dict_all_init = eval(data)
        except:
            raise Exception("text in file not properly formatted dictionary")
        else:
            # Make it robust to errors here
            ...

        self.target_scorecount = target_scorecount

        # Keep track of section ids, average scores, LO names, globally
        self.section_ids = list(dict_all_init.keys())

        if anonymize:
            # Shuffle key_order to anonymize report
            key_order = list(dict_all_init.keys())
            rnd.shuffle(key_order)
            self.dict_all = {k : dict_all_init[k] for k in key_order}

            self.section_names = [f'Section {chr(i+65)}' for i in range(len(self.section_ids))]
        else:
            self.section_names = [f'Section {chr(i+65)}' for i in range(len(self.section_ids))]
            self.dict_all = dict_all_init

        # Need to re-set section_ids so that they're in the same order as the potentially shuffled key order
        self.section_ids = list(self.dict_all.keys())

        self.all_scores = []
        self.all_LOs = set()

        self.section_ids_w_scores = copy.deepcopy(self.section_ids)
        for section_id in self.section_ids:
            # A sublist for this section
            self.all_scores.append([])
            for student_id in self.dict_all[section_id]:
                # A sublist for this student
                self.all_scores[-1].append([])
                for submission_data in self.dict_all[section_id][student_id]:
                    if submission_data['score'] != None:
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
            print("This section scores", self.all_scores[i])

        print("all_scores:", self.all_scores)
        print("all_avgscores:", self.all_avgscores)
        print("All sections with something", self.section_ids)
        print("All sections with scores", self.section_ids_w_scores)


        
        # Set up the colors for the different sections
        self.section_colors = disc_colors.Dark24[:len(self.section_ids)]

        # set up section colors for the tables
        # Convert hexadecimal representation to rgb
        table_section_colors = [tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4)) for color in self.section_colors]
        # Increase brightness with 20%
        table_section_colors = [(min(255, int(rgb_val[0]*1.2)), min(255, int(rgb_val[1]*1.2)), min(255, int(rgb_val[2]*1.2))) for rgb_val in table_section_colors]
        # Convert to string and set alpha=0.6
        self.table_section_colors = [f'rgba({r}, {g}, {b}, 0.6)' for r, g, b in table_section_colors]

        
        # The list of all figures (or html text), in the right order, to be included in the report html
        self.figures = []
    

    def progress_table(self) -> None:
        ''' Produces table with progress indication, adds it to report '''

        # Count number of scores and comments
        scores_counts = []
        comments_counts = []
        comment_wordcounts = []
        for section_id in self.section_ids:
            score_counter = 0
            comment_counter = 0
            comment_wordcounter = 0
            student_count = 0
            print(self.dict_all[section_id].keys())
            for student_id in self.dict_all[section_id].keys():
                student_count += 1
                for submission_data in self.dict_all[section_id][student_id]:
                    if submission_data['score'] is not None:
                        score_counter += 1
                    if len(submission_data['comment']) > 1:
                        comment_counter += 1
                        comment_wordcounter += len(submission_data['comment'].split(' '))
            scores_counts.append(round(score_counter/student_count,2))
            comments_counts.append(round(comment_counter/student_count,2))
            comment_wordcounts.append(round(comment_wordcounter/student_count,2))

        # Setup colors for score count
        if self.target_scorecount is None:
            scores_colorscale = ['lightblue']
            score_color_indices = [0] * len(scores_counts)
        else:
            num_colors = 5
            scores_colorscale = sample_colorscale('RdYlGn', list(np.linspace(0.15, 0.85, num_colors)))
            scores_colorscale
            score_color_indices = [min(num_colors-1, int((num_colors-1)*score_count/self.target_scorecount)) for score_count in scores_counts]

        # Manually set colors in table to display them, remove this later
        #score_color_indices[0] = 0
        #score_color_indices[1] = 1
        #score_color_indices[2] = 2
        #score_color_indices[3] = 3
        #score_color_indices[4] = 4

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
                    y = 1.15,
                    x = 1
                )],
                height=270+29*len(self.section_ids),# * (1 if self.anonymize else 2),
                font_size=15)
        
        self.figures.append('<center><h1>Grading progress</h1></center>')
        self.figures.append("<i>The key associating each section ('Section C') with it's Forum section ID ('16193482') is found at the bottom of the report</i>")
        
        self.figures.append(fig)

    def LO_progress_table(self) -> None:
        ''' Produces table with progress indication, adds it to report '''

        # Count number of scores and comments
        lo_scores_counts = [[0 for _ in range(len(self.section_ids))] for _ in range(len(self.all_LOs))]
        for lo_index, lo_name in enumerate(self.all_LOs):
            for section_index, section_id in enumerate(self.section_ids):
                for student_id in self.dict_all[section_id].keys():
                    for submission_data in self.dict_all[section_id][student_id]:
                        if submission_data['learning_outcome'] == lo_name:
                            lo_scores_counts[lo_index][section_index] += 1

        # Setup colors for score count
        #if self.target_scorecount is None:
        #    scores_colorscale = ['lightblue']
        #    score_color_indices = [0] * len(scores_counts)
        #else:
        #    num_colors = 5
        #    scores_colorscale = sample_colorscale('RdYlGn', list(np.linspace(0.15, 0.85, num_colors)))
        #    scores_colorscale
        #    score_color_indices = [min(num_colors-1, int((num_colors-1)*score_count/self.target_scorecount)) for score_count in scores_counts]

        # Calculate total number of scores per section
        total_scores_given = np.array(lo_scores_counts[0])
        for lo_index in range(len(lo_scores_counts)-1):
            total_scores_given += np.array(lo_scores_counts[lo_index])

        greys_colorscale = sample_colorscale('Greys', list(np.linspace(0, 1, 101)))

        # Figure out what color to assign each cell for each LO, in order of LO appearance in self.all_LOs
        lo_color_indices = [[] for _ in range(len(self.all_LOs))]
        for lo_index in range(len(self.all_LOs)):

            # If the max count equals the min count, display the average color for all
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
        print("lo_color_indices: ", lo_color_indices)
        # Collect all columns in one list
        data = [self.section_names, total_scores_given]
        data.extend(lo_scores_counts)

        column_names = ['Section name', 'Total scores assigned']
        column_names.extend(['Score count in ' + lo_name for lo_name in self.all_LOs])
        
        column_colors = [self.table_section_colors, np.array(greys_colorscale)[tot_lo_color_indices]]
        tot_lo_colors = np.array(greys_colorscale)[tot_lo_color_indices]
        print(tot_lo_color_indices)
        print(tot_lo_colors)
        print(len(column_colors))
        column_colors.extend([np.array(greys_colorscale)[lo_color_indices[i]] for i in range(len(self.all_LOs))])
        print(len(column_colors))

        print("\n\n\n\n\n\n\n\n")
        print("values:", ['Section name'].extend(self.all_LOs))
        print("data:", data)
        print("column colors:")
        for color_lst in column_colors:
            print(color_lst)
        
        # Create table figure, with appropriate colors
        fig = go.Figure(data=[go.Table(header=dict(values=column_names),
                                        cells=dict(values=data,
                                                    fill_color=column_colors,
                                                    height=30))])
        
        # Make a dataframe of the data, it's more easily sortable
        df = pd.DataFrame(data).T.sort_values(0).T

        # Create dictionaries for all colors, with section_id as key, to be able to maintain colors after sorting
        #section_color_dict = {section_id:self.table_section_colors[i] for i, section_id in enumerate(self.section_names)}
        #total_scores_color_dict = {section_id:column_colors[1][i] for i, section_id in enumerate(self.section_names)}
        #print("column colors length:")
        #print(len(column_colors))
        #for color_lst in column_colors:
        #    print(color_lst)
        #score_count_color_dicts = [{section_id:column_colors[lo_index+2][i] for i, section_id in enumerate(self.section_names)} for lo_index in range(len(self.all_LOs))]
        #scores_color_dict = {section_id:scores_colorscale[score_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        #comm_color_dict = {section_id:greys_colorscale[comm_count_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        #word_color_dict = {section_id:greys_colorscale[word_count_color_indices[i]] for i, section_id in enumerate(self.section_names)}


        #menu_selection = [{"name": "Sort by section name", "col_i": 0}, 
        #                  {"name": "Sort by total scores", "col_i": 1}]
        #menu_selection.extend([{"name": f"Sort by {lo_name}", "col_i": 2+i} for i, lo_name in enumerate(self.all_LOs)])
        ## Create sorting drop-down menu
        fig.update_layout(
        #    updatemenus=[dict(
        #            buttons= [dict(
        #                    method= "restyle",
        #                    label= selection["name"],
        #                    args= [{"cells": {"values": df.T.sort_values(selection["col_i"]).T.values, # Sort all values according to selection
        #                                        "fill": dict(color=[[section_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"]).T.values[0]], # Ensure all colors are with the correct cell
        #                                                [total_scores_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"]).T.values[0]],
        #                                                *[[score_count_color_dicts[lo_index][section_id] for section_id in df.T.sort_values(selection["col_i"]).T.values[0]] for lo_index in self.all_LOs]
        #                                                ]),
        #                                        "height": 30}},[0]]
        #                    )
        #                    for selection in menu_selection
        #            ],
        #            direction = "down",
        #            y = 1.15,
        #            x = 1
        #        )],
                height=300+29*len(self.section_ids),# * (1 if self.anonymize else 2),
                font_size=15)
        
        self.figures.append('<center><h1>LO grading progress</h1></center>')
        self.figures.append('<center><i>work in progress, this table will be prettier</i></center>')
        
        self.figures.append(fig)

    def summary_stats_table(self) -> None:
        ''' Produces table with summary statistics, adds it to report '''
        # Calculate section means and SDs
        self.section_means = np.array([np.mean(section_avgs) for section_avgs in self.all_avgscores])
        self.section_SDs = np.array([np.std(section_avgs) if len(section_avgs) > 0 else np.nan for section_avgs in self.all_avgscores])

        print("section means:", self.section_means)
        print("section SDs:", self.section_SDs)

        # Number of students in each section
        self.section_sizes = np.array([len(self.dict_all[section_id].keys()) for section_id in self.section_ids])

        # Overall average score and SDs, weighted by number of students. This defines the "average" section
        overall_mean = sum(self.section_means * self.section_sizes)/sum(self.section_sizes)
        overall_SD = sum(self.section_SDs * self.section_sizes)/sum(self.section_sizes)
        average_section_size = np.mean(self.section_sizes)

        p_values = []
        effect_sizes = []
        # Perform two-tailed difference of means test
        # between each section and the "average" section
        for i in range(len(self.section_means)):
            # Standard error
            standard_err = np.sqrt(self.section_SDs[i]**2 / self.section_sizes[i] + \
                                   overall_SD**2 / average_section_size)
            # Effect size, and t-statistic
            effect_sizes.append((self.section_means[i] - overall_mean)/overall_SD)
            t_stat = (self.section_means[i] - overall_mean)/standard_err
            
            # degrees of freedom
            df = min(average_section_size-1, self.section_sizes[i]-1)
            
            p_values.append(sts.t.sf(abs(t_stat), df) * 2)

        # Assign colors to cells

        # Create colorscales
        redblue_colorscale = sample_colorscale('RdYlBu', list(np.linspace(0, 1, 101)))
        greys_colorscale = sample_colorscale('Greys', list(np.linspace(0, 1, 101)))

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
                    mean_color_indices.append(int(15+70*(mean_val-min(self.section_means))/(max(self.section_means)-min(self.section_means))))
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
        min_abs_effect = min([abs(val) for val in effect_sizes]) 
        max_abs_effect = max([abs(val) for val in effect_sizes])
        
        # If all effect sizes are equal
        print("Effect sizes", effect_sizes)
        if max_abs_effect - min_abs_effect == 0 or np.isnan(effect_sizes).any() or math.isinf(max_abs_effect):
            effect_color_indices = [10] * len(effect_sizes)
        else:
            # Let effect size vary linearly from 0.0 to 0.7 on yellowred colorscale
            effect_color_indices = [int(10+40*(abs(val)-min_abs_effect)/(max_abs_effect-min_abs_effect)) for val in effect_sizes]

        # Manual logic for coloring the p-values
        p_colors = []
        for val in p_values:
            if val < 0.05: #0.05
                # significant
                p_colors.append('red')
            else:
                # Not significant
                p_colors.append('white') 

            #elif val < 0.1: #0.1
            #    # close to significant
            #    p_colors.append('orange')
            #elif val < 0.3: #0.3
            #    # not strongly non-significant
            #    p_colors.append('mistyrose') 
            #else:
            #    # strongly not significant
            #    p_colors.append('palegreen')
                
        # Create table
        data = [self.section_names,
                [round(val,3) for val in self.section_means],
                [round(val,3) for val in self.section_SDs],
                [round(val,5) for val in p_values],
                [round(val,5) for val in effect_sizes]]

        fig = go.Figure(data=[go.Table(header=dict(values=
                                                    ['Section name', 
                                                     'Mean score',
                                                     'Standard Deviation',
                                                     'p-values',
                                                     'effect size']),
                                        cells=dict(values=data,
                                                    fill_color=[
                                                        self.table_section_colors,
                                                        np.array(redblue_colorscale)[mean_color_indices],
                                                        np.array(greys_colorscale)[sd_color_indices],
                                                        p_colors,
                                                        np.array(greys_colorscale)[effect_color_indices]
                                                    ],
                                                    height=30))])
        
        # Make a dataframe of the data, it's more easily sortable
        df = pd.DataFrame(data).T.sort_values(0).T

        # Create dictionaries for all colors, with section_id as key, to be able to maintain colors after sorting
        section_color_dict = {section_id:self.table_section_colors[i] for i, section_id in enumerate(self.section_names)}
        mean_color_dict = {section_id:redblue_colorscale[mean_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        sd_color_dict = {section_id:greys_colorscale[sd_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        p_color_dict = {section_id:p_colors[i] for i, section_id in enumerate(self.section_names)}
        effect_color_dict = {section_id:greys_colorscale[effect_color_indices[i]] for i, section_id in enumerate(self.section_names)}
        

        # Create sorting drop-down menu
        fig.update_layout(
            updatemenus=[dict(
                    buttons= [dict(
                            method= "restyle",
                            label= selection["name"],
                                                                                            # Sort ascending only for column 0 and 3
                            args= [{"cells": {"values": df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0 or selection["col_i"]==3).T.values, # Sort all values according to selected column
                                                "fill": dict(color=[[section_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0 or selection["col_i"]==3).T.values[0]], # Ensure colors are with correct cell
                                                        [mean_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0 or selection["col_i"]==3).T.values[0]],
                                                        [sd_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0 or selection["col_i"]==3).T.values[0]],
                                                        [p_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0 or selection["col_i"]==3).T.values[0]],
                                                        [effect_color_dict[section_id] for section_id in df.T.sort_values(selection["col_i"], ascending=selection["col_i"]==0 or selection["col_i"]==3).T.values[0]]
                                                        ]), 
                                                "height": 30}},[0]]
                            )
                            for selection in [{"name": "Sort by section name", "col_i": 0}, 
                                      {"name": "Sort by mean", "col_i": 1}, 
                                      {"name": "Sort by standard deviation", "col_i": 2}, 
                                      {"name": "Sort by p-value", "col_i": 3}, 
                                      {"name": "Sort by effect size", "col_i": 4}]
                    ],
                    direction = "down",
                    y = 1.15,
                    x = 1
                )],
                height=250+29*len(self.section_ids),# * (1 if self.anonymize else 2),
                font_size=15)

        self.figures.append('<center><h1>Summary statistics</h1></center>')
        self.figures.append('p-values and effect size are from a difference of means test, <br>comparing each section with the "average" section. <br>This calculation is currently quite basic and will be improved.')
        
        self.figures.append(fig)

    def section_avg_scores(self, section_id: int) -> list:
        ''' Returns a list with the average score of each student for this section '''
        
        student_ids = list(self.dict_all[section_id].keys())
        avg_scores = []

        # For each student
        for this_student_id in student_ids:
            # Calculate average score
            student_scores = []
            for score_data in self.dict_all[section_id][this_student_id]:
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

        # For each section
        print("\n\n\n\n\n all avgscores", self.all_avgscores)
        all_scores_flat = [avg_score for section_lst in self.all_avgscores for avg_score in section_lst]
        
        fig = go.Figure(data=go.Histogram(
                    x=all_scores_flat,
                    xbins=dict(
                        start=0,
                        end=5,
                        size=bin_size
                    ),
                    opacity=0.8))
        # Make histogram
        
        
        fig.update_traces(marker_line_width=1,marker_line_color="white")
        fig.update_layout(
            title=f'Score distribution across all sections ({len(gd.section_ids)} sections)',
            xaxis_title='Score',
            yaxis_title='Count',
            barmode='overlay')
        
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

        print("ANOVA's scores", scores)
        print("self.all_avgscores", self.all_avgscores)
        # Perform the ANOVA test
        try:
            # Remove all nan from scores
            filtered_scores = []
            for sublist in scores:
                print("sub:", sublist)
                if not np.isnan(sublist[0][0]):
                    filtered_scores.append([val for lst in sublist for val in lst])
            print("filtered scores!:")
            print(filtered_scores)


            f_stat, p_value = sts.f_oneway(*filtered_scores)
            #f_stat, p_value = f_stat[0], p_value[0]
            print("Anova stuff")
            print(f_stat)
            print(p_value)
            if np.isnan(f_stat) or np.isnan(p_value):
                raise Exception("Not enough data")

            output = ''
            # Display the results
            # Small p-value means statistically significant
            output += "<center><h1> ANOVA Results</h1></center>"
            output += "<h3>P-value: " + str(round(p_value, 3)) + "</h3>"
            output += "(With significance level = 0.05)<br>"
            if p_value < 0.05:
                output += "We reject the null hypothesis. At least one section has a different mean than the other groups, with statistical significance. There's a " + str(round(p_value*100, 2)) + "% chance of a Type I error.\n\n"
            else:
                output += "We don't reject the null hypothesis. We don't have statistically significant evidence that any group has a mean different from the others.\n\n"
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

        # Flatten the scores, now including None
        flat_scores = [score for lst in scores for score in lst]
        

        # x_mapping describes which datapoint belongs to which section
        x_mapping = []
        for i, section_name in enumerate(self.section_names):
            x_mapping.extend([section_name] * len(scores[i]))

        
        # Boxplot for scores, displaying quartiles
        #fig.add_trace(go.Box(
        #    y=flat_scores,
        #    x=x_mapping,
        #    name='4 Quartiles',
        #    quartilemethod="inclusive",
        #    fillcolor=self.section_colors[i],
        #    line=dict(color='black') # self.section_colors
        #))

        # For a given section in this list,
        # All datapoints will be at the mean except
        # 2 datapoints which will be +1 and -1 SD.
        means_and_SDs = []

        # calcuate means and SDs
        for section_scores in scores:
            filtered_list = [score for score in section_scores if score is not None]
            # If there's any scores in this section
            if len(filtered_list) > 0:
                mean = np.mean(filtered_list)
                std_dev = np.std(filtered_list)
                means_and_SDs.append(mean-std_dev)
                means_and_SDs.append(mean+std_dev)

                # Fill the rest of the section with the mean
                remaining_count = len(section_scores) - 2
                means_and_SDs.extend([mean] * remaining_count)
            # There are no scores in this section yet
            else:
                means_and_SDs.extend([None]*5)

        for i, group in enumerate(self.section_names):
            lst = [y_val for y_val, x_val in zip(flat_scores, x_mapping) if x_val == group]
            x_lst = [[group] * len(scores[i]),['4 Quartiles'] * len(scores[i])]
            fig.add_trace(go.Box(
                y=scores[i],
                x=x_lst,
                #x=[i-0.2] * len(lst),
                name=group,
                marker_color=self.section_colors[i],
                #name='4 Quartiles',
                legendgroup=group,
                #legendgroup='4 Quartiles',
                #legendgrouptitle_text=group,
                quartilemethod="inclusive",
                showlegend = True
            ))

            lst = [y_val for y_val, x_val in zip(means_and_SDs, x_mapping) if x_val == group]
            x_lst_a = ['4 Quartiles', 'Mean & ±1 SD'] * len(self.section_names)
            x_lst_b = [[name] * 2 for name in self.section_names]
            x_lst = [x_lst_a, x_lst_b]

            x_lst = [[group] * 2, ['4 Quartiles', 'Mean & ±1 SD']]

            x_lst = [[group] * len(scores[i]), ['Mean & ±1 SD'] * len(scores[i])]
            # Boxplot only displaying mean and SDs
            fig.add_trace(go.Box(
                y=lst,
                x=x_lst,
                #x=[i+0.2] * len(lst),
                name=group,
                #name='Mean & ±1 SD',
                #legendgroup='Mean & ±1 SD',
                legendgroup=group,
                #legendgrouptitle_text=group,
                marker_color='black', 
                quartilemethod="inclusive",
                boxpoints=False,
                showlegend = False
            )) 



        fig.update_layout(
            title='<b>Distribution of average score per section</b><br>Box plot, colored boxplots are quartiles, black lines are mean +- 1 SD',
            xaxis_title='Section',
            yaxis_title='Scores',
            height=800,
            legend=dict(groupclick="togglegroup")
            #boxmode='group'
        )


        with open('only_boxplot.html', 'a') as f:
            # Remove contents
            f.truncate(0)
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
            for student_id in self.dict_all[section_id]:
                for submission_data in self.dict_all[section_id][student_id]:
                    if submission_data['learning_outcome'] == LO_name:
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
            for student_id in self.dict_all[section_id]:
                for submission_data in self.dict_all[section_id][student_id]:
                    if submission_data['learning_outcome'] == LO_name:
                        # Increment the the counter corresponding to the score
                        LO_scores_count[-1][int(submission_data['score'])-1] += 1
        
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
            title=f'<b>Percentage of students receiving score in {LO_name} per section</b><br>Stacked barplots.',
            xaxis_title='Section',
            yaxis_title='Percentage',
            height=800,
            barmode='stack')
        
        fig.layout.yaxis.tickformat = '0%'

        # Add plot to the report
        self.figures.append(fig)
        

    def section_id_table(self) -> None:
        ''' Add a small table associated anonymous labels of sections to their true section ID '''

        fig = go.Figure(data=[go.Table(header=dict(values=
                                                    ['Section Name', 
                                                     'Section ID']),
                                        cells=dict(values=[self.section_names, self.section_ids],
                                                    fill_color=[
                                                        self.table_section_colors
                                                    ],
                                                    height=30))])
        
        fig.update_layout(height=250+35*len(self.section_ids), font_size=15)

        self.figures.append('<center><h1>Section name to ID key</h1></center>')
        
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
            </style>
        </head>
        <body>
        '''

        html_end = "</body>"

        self.figures.insert(0, html_head)
        self.figures.append(html_end)

        with open('grading_dashboard.html', 'a') as f:
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
                for student_id in self.dict_all[section_id]:
                    for submission_data in self.dict_all[section_id][student_id]:
                        if submission_data['learning_outcome'] == lo_name:
                            score_count += 1
            lo_score_counts.append(score_count)

        # Sort the lo names according to the number of scores
        return [LO_name for _, LO_name in sorted(zip(lo_score_counts, self.all_LOs), key=lambda pair: pair[0], reverse=True)]


    def make_full_report(self) -> None:
        ''' Creates a pre-selected set of plots and results, in the right order, then creates html '''

        self.ANOVA_test(False)
        self.progress_table()
        self.LO_progress_table()
        self.summary_stats_table()
        self.scoreavgs_allsections_plot()
        self.figures.append("<center><h1>Student average score distributions</h1></center>")
        self.figures.append("- Click and drag inside the plot to zoom in, double click to reset<br>")
        self.figures.append("- Click the legend on the right to select and deselect different sections")
        self.boxplots()
        # Skip the violinplots (kde)
        # self.violinplots()
        self.figures.append("<center><h1>LO score distributions</h1></center>")

        sorted_LOs = self.get_sorted_LOs()

        for lo_name in sorted_LOs:
            self.LO_stackedbar_plot(lo_name)

        self.section_id_table()
        self.create_html()


import json
# Create new file with only portions of fake_data
def create_data(file_name:str, total_scores:int):

    # Read in data

    # Open the file
    with open(file_name, 'r', encoding='utf8') as file:
        data = file.read()

    # Dangerous eval call, should test for errors
    try:
        dict_all = eval(data)
    except:
        raise Exception("text in file not properly formatted dictionary")

    # Edit data in dict variable
    all_scores_path = []

    # Get all score paths
    for section_ID in dict_all.keys():
        for student_ID in dict_all[section_ID].keys():
            for score_index in range(len(dict_all[section_ID][student_ID])):
                all_scores_path.append((section_ID, student_ID, score_index))
        
    rnd.seed(10)
    chosen_scores = rnd.sample(all_scores_path, total_scores)
    #print(chosen_scores)
    new_dict = {}

    for score_path in chosen_scores:
        if score_path[0] in new_dict.keys():
            if score_path[1] in new_dict[score_path[0]].keys():
                # Add the score data to the list, if the list already exists
                new_dict[score_path[0]][score_path[1]].append(dict_all[score_path[0]][score_path[1]][score_path[2]])
            else:
                # This student doesn't exist yet, add it with a list containing the score data
                new_dict[score_path[0]][score_path[1]] = [dict_all[score_path[0]][score_path[1]][score_path[2]]]
        else:
            # This section doesn't exist yet, add it with the correct student and data list
            new_dict[score_path[0]] = {score_path[1]: [dict_all[score_path[0]][score_path[1]][score_path[2]]]}

    # Write dictionary into new file
    new_file_name = 'fake_data_new.py'

    with open(new_file_name, 'w', encoding='utf8') as file:
        file.write(str(new_dict))
        

    return new_file_name

new_data_name = create_data('fake_data_986100.py', 28)
gd = GradingDashboard('fake_data_986100.py', anonymize=False, target_scorecount=6)


gd.make_full_report()

import os
os.system("start file:///C:/Users/gabri/Desktop/Grading%20Dashboard/grading_dashboard.html")
