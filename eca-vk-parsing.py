# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:01:04 2020

@author: Artem

PARSE USERS FROM ECA GROUPS IN PODOLSK

"""

# Load libraries
import os
import re
import time
import math
import pandas as pd
import numpy as np
import vk_api
from itertools import combinations_with_replacement
from vk_api import exceptions
from vk_api.execute import VkFunction
from vk_api.requests_pool import vk_request_one_param_pool
import seaborn as sns
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Networks
import networkx as nx

# Change the working directory
os.chdir('C:/Users/Artem/Google Drive/WB/ECA_2019/Vkontakte_data/working_files')

# Set vk session
vk_session = vk_api.VkApi('VK_API_LOGIN', 'VK_API_PASSWORD')
vk_session.auth()
vk = vk_session.get_api()

# Load the starting groups
eca_groups = pd.read_excel("eca_info_podolsk.xlsx")
eca_groups["org_type"] = "ECA"
school_groups = pd.read_excel("schools_info_podolsk.xlsx")
school_groups['inn'] = school_groups['inn'].apply(lambda x: str(x))
school_groups['inn_name'] = school_groups['inn'] + "_" + school_groups['name']
school_groups = school_groups.loc[:,["inn_name", "soc_media", "org_type"]]
school_groups.columns = ["inn_name", "vk_link", "org_type"]

# Merge groups together
start_groups = pd.concat([eca_groups, school_groups])

# ! IMPORTANT Remove schools
start_groups = start_groups[start_groups['org_type'] == "ECA"]

# Parse links to VK groups from the social media column

'''
def getVkLink(links):
    result = np.nan
    if type(links) == str:
        temp_split = links.split(', ')
        result = list(filter(lambda url: 'vk.com' in url, temp_split))
        if len(result) > 0:
            result = result[0]
        else:
            result = np.nan
    return result

start_groups['vk_link'] = start_groups['soc_media'].apply(getVkLink)
'''

# Create separate dataframe with only organization with VK groups 
start_groups_vk = start_groups[start_groups['vk_link'].notna()]
# start_groups_vk = start_groups_vk[start_groups.own_type == 'state']

# Remove https://vk.com/ from the beginning
start_groups_vk['vk_screen_name'] = start_groups_vk['vk_link'].apply(lambda link: link.split('/')[-1])


def removeClubPublic(string):
    '''
    Remove 'club' and 'public' at the beginning of a string
    '''
    result_string = string
    if re.search('^club', string):
        if re.search('^club[^0-9]', string):
            pass
        else:
            result_string = re.sub('^club', '', string)
    if re.search('^public', string):
        if re.search('^public[^0-9]', string):
            pass
        else:
            result_string = re.sub('^public', '', string)
    
    return result_string


# Remove 'club' and 'public' at the beginning of a string
start_groups_vk['vk_screen_name'] = start_groups_vk['vk_screen_name'].apply(removeClubPublic)

# Drop duplicates based on Vk screen name
start_groups_vk = start_groups_vk.drop_duplicates('vk_screen_name')

# Get groups information
def chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getGroupsInfo(group_ids_list):
    '''
    Parse full information of a group:
        id, screen name, full name, members count
    '''

   # Split list of users ids by chunks
    groups_ids_chunks = list(chunks(group_ids_list, 100))
    
    groups_info = pd.DataFrame()
    chunks_completed = 0
    for chunk in groups_ids_chunks:
    
        groups_pool = vk_request_one_param_pool(vk_session = vk, 
                                                method = 'groups.getById',
                                                default_values = {'fields':'members_count'},
                                                key = 'group_ids',
                                                values = chunk)
        
        groups_pool = pd.DataFrame(groups_pool[0]).T
        groups_pool = groups_pool[0].apply(pd.Series)
        groups_pool = groups_pool.loc[:,['id','screen_name', 'name', 'members_count', 'is_closed']]
        
        groups_info = groups_info.append(groups_pool)
        
        chunks_completed += len(chunk)
        print(np.round((chunks_completed / len(group_ids_list)) * 100, 2))
    
    return groups_info


# Get full information about groups
start_groups_info = getGroupsInfo(list(start_groups_vk['vk_screen_name']))
start_groups_info.reset_index(inplace=True)

# Add numerical id to intital dataframe. Remove ECA organizations with closed VK groups
start_groups_vk = start_groups_vk.merge(start_groups_info,
                                        left_on='vk_screen_name', right_on='index')


def getUsersFromGroup(group_id):
    '''
    Parse all users from one group.
    Gets as an input ID of a group.
    Returns list of user IDs in a group.
    '''
    group_members = []
    
    try:
        members_count = vk.groups.getById(group_id=group_id, fields={'members_count'})[0]['members_count']
    
        if members_count <= 1000:
            try:
                group_members.extend(vk.groups.getMembers(group_id=group_id)['items'])
                group_members = list(map(lambda member_id: str(member_id), group_members))
            except exceptions.ApiError:
                pass
            
        else:
            for i in range(math.ceil(members_count / 25000)):
                vk_script =  'var members = API.groups.getMembers({"group_id":  %(group_id)s, "count": "1000", "offset": ' + '{}'.format(len(group_members)) + '}).items;' +\
                        'var offset = 1000;' +\
                        'while (offset < 25000 && (offset + ' + '{}'.format(len(group_members)) + ') < ' + '{}'.format(members_count) + ')'  +\
                        '{' +\
                            'members = members + "," + API.groups.getMembers({"group_id": %(group_id)s, "count": "1000", "offset": (' + '{}'.format(len(group_members)) + ' + offset)}).items;' +\
                            'offset = offset + 1000;' +\
                        '};' +\
                        'var members = members.split(",");' +\
                        'return members;'
                vk_exe_get_groups = VkFunction(args=('group_id',), code = vk_script)
                try:
                    group_list = vk_exe_get_groups(vk, {'group_id': group_id})
                    group_members.extend(group_list)
                except exceptions.ApiError:
                    pass
                time.sleep(0.3)
    except:
        pass

    return group_members

# Get list of users 
df_users_groups = []
for i in range(len(start_groups_vk['id'])):
    df_users_groups.append(getUsersFromGroup(start_groups_vk['id'].iloc[i]))
    clear_output(wait=True)
    print(i)
    
df_users_groups = pd.Series(df_users_groups)
df_users_groups = pd.DataFrame({'group_id': start_groups_vk['id'],
                                'user_ids': list(df_users_groups)})

# Remove closed groups
df_users_groups = df_users_groups[df_users_groups['user_ids'].apply(
        lambda group_id: False if len(group_id) == 0 or group_id[0] == '' else True)]

# Remove duplicates
df_users_groups.drop_duplicates('group_id', inplace=True)

# Save to pickle
df_users_groups.to_pickle('df_users_groups_eca_schools_podolsk.obj')
df_users_groups = pd.read_pickle('df_users_groups_eca_schools_podolsk.obj')

# Matrix of pairwise intersection
def pairwise(X, operation):
    '''
    Compute matrix of parwise intersection between groups
    '''
    # Initialise precomputed matrix
    # precomputed = np.zeros((X.shape[0], X.shape[0]), dtype='int')
    precomputed = np.zeros((X.shape[0], X.shape[0]), dtype='float')
    # Initialise iterator over objects in X
    iterator = combinations_with_replacement(range(X.shape[0]), 2)
    # Perform the operation on each pair
    for i, j in iterator:
        precomputed[i, j] = operation(X[i], X[j])           
    # Make symmetric and return
    return precomputed + precomputed.T - np.diag(np.diag(precomputed))


def overlap(x, y):
    '''
    Calculate intersection between two groups
    '''
    return len(set(x) & set(y))


def overlapNormalized(x, y):
    '''
    Calculate normalized intersection between two groups
    '''
    return len(set(x) & set(y)) / max(len(set(x)), len(set(y)))


# Create matrix of pairwise intersection
user_ids_list = np.array(df_users_groups['user_ids'])
matrix_intersections = np.matrix(pairwise(user_ids_list, overlap))
np.fill_diagonal(matrix_intersections, 0)
matrix_intersections = pd.DataFrame(matrix_intersections)

# Set names for columns and rows
matrix_intersections.columns = list(df_users_groups['group_id'])
matrix_intersections.index = list(df_users_groups['group_id'])

# Create edgelist
edge_list = matrix_intersections.astype(float)
edge_list.values[[np.arange(len(edge_list))]*2] = np.nan
edge_list = edge_list.stack().reset_index()
# Create labels for edgelist dataframe
edge_list.columns = ['group_id_1', 'group_id_2', 'weight']
# Convert group id to int
# edge_list = edge_list.astype(int)
edge_list['group_id_1'] = edge_list['group_id_1'].astype(int)
edge_list['group_id_2'] = edge_list['group_id_2'].astype(int)

# Drop duplicated edges
edge_list['pair'] = edge_list.loc[:,['group_id_1', 'group_id_2']].apply(lambda row: sorted([row[0], row[1]]), 1)
edge_list['pair'] = edge_list['pair'].apply(lambda row: "_".join([str(row[0]), str(row[1])]))
edge_list = edge_list.drop_duplicates("pair")
edge_list = edge_list.drop(columns="pair")

edge_list = edge_list[edge_list['weight'] >= 1]

# Write edgelist to txt file
outF = open("eca_schools_podolsk_groups_edgelist.txt", "w")
for i in range(len(edge_list)):
  # write line to output file
  outF.write(str(int(edge_list.iloc[i][0])) + ' ' + str(int(edge_list.iloc[i][1])) + ' ' + str(int(edge_list.iloc[i][2])))
  outF.write("\n")
outF.close()
