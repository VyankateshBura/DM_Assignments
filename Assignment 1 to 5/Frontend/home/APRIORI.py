import pandas as pd
import numpy as np
import math


def generate_candidate_itemsets(itemset, k):
    candidate_itemset = set()
    for item1 in itemset.keys():
        for item2 in itemset.keys():
            if len(item1.union(item2)) == k:
                candidate_itemset.add(item1.union(item2))
    
    return candidate_itemset

def get_frequent_itemsets(data, min_support,column_names):
    data = data.drop("Class Name", axis=1)
    itemset = {frozenset({item}): 0 for item in column_names[1:]}
    for index, row in data.iterrows():
        i=0
        for frozen_key in itemset.keys():
            if row[i]==1:
                itemset[frozen_key]+=1
            i+=1

    num_items = float(len(data))

    frequent_itemset = {item: support / num_items for item, support in itemset.items() if support / num_items >= min_support}
    return frequent_itemset

def APRIORI(data, min_support, max_len,column_names):

    try:
        frequent_itemsets = []
        k = 0
        dataset = []
        for index, row in data.iterrows():
            i=0
            lt = set()
            for col in column_names[1:]:
                if row[col]==1:
                    lt.add(col) 
            dataset.append(lt)

        while True:
            if k == 0:
                frequent_itemsets.append(get_frequent_itemsets(data, min_support,column_names))
 
            else:
                if not frequent_itemsets or not frequent_itemsets[k - 1]:
                    break

                last_frequent_itemset = frequent_itemsets[k - 1]
                candidate_itemsets = generate_candidate_itemsets(last_frequent_itemset, k)
                
                frequent_itemset = {}
                for candidate in candidate_itemsets:
                    count = 0
                    for transaction in dataset:
                        if candidate.issubset(transaction):
                            count += 1
                    frequent_itemset[candidate] = count
                    
                frequent_itemset = {item: support /len(dataset) for item, support in frequent_itemset.items() if support /len(dataset) >= min_support}
                if not frequent_itemset:
                    break

                frequent_itemsets.append(frequent_itemset)

            k += 1
            if k > max_len:
                break

        return frequent_itemsets
    except Exception as e:
        print(e)
        print("Error in AprioriAlgo")



def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            for item in itemset:
                antecedent = frozenset({item})
                # print(item,type(itemset),type(antecedent))
                consequent = frozenset(itemset - antecedent)
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                lift = confidence/frequent_itemsets[consequent]
                

                if confidence >= min_confidence:
                    rules.append((list(antecedent), list(consequent), confidence,lift))


    return rules




