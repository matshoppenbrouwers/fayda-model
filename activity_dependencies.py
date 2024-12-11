from collections import defaultdict
import re
import networkx as nx
import pandas as pd
import numpy as np
import logging

class ActivityDependencyAnalyzer:
    def __init__(self):
        self.dependency_rules = [
            # Input/Output dependencies
            (r'receive|get|request', r'process|validate|scan|enter'),
            (r'scan|enter|input', r'validate|verify|check'),
            (r'generate|create', r'issue|prepare'),
            
            # Process flow dependencies
            (r'open|start|initiate', r'process|validate|complete'),
            (r'validate|verify', r'approve|confirm|generate'),
            (r'prepare|generate', r'issue|send|deliver'),
            
            # Document handling dependencies
            (r'scan', r'attach|upload|store'),
            (r'collect|gather', r'enter|input|record'),
            
            # Client interaction dependencies
            (r'ask|request', r'receive|collect|record'),
            (r'inform|explain', r'get|obtain|receive'),
        ]

    def analyze_dependencies(self, activities):
        """Analyze dependencies between activities based on naming patterns and business rules"""
        G = nx.DiGraph()
        
        # Add all activities as nodes
        for _, row in activities.iterrows():
            G.add_node(row['Activity Name'])
        
        # Analyze pairs of activities for dependencies
        for i, row1 in activities.iterrows():
            name1 = row1['Activity Name'].lower()
            
            for j, row2 in activities.iterrows():
                if i == j:
                    continue
                    
                name2 = row2['Activity Name'].lower()
                
                # Check for dependencies based on rules
                for pattern1, pattern2 in self.dependency_rules:
                    if (re.search(pattern1, name1) and re.search(pattern2, name2)):
                        G.add_edge(row1['Activity Name'], row2['Activity Name'])
                        logging.debug(f"Found dependency: {row1['Activity Name']} -> {row2['Activity Name']}")
        
        return G

    def get_processing_batches(self, G):
        """Group activities into batches that can be processed together"""
        try:
            # Get levels using longest path length
            levels = defaultdict(list)
            for node in G.nodes():
                # Find the longest path length to this node
                paths = [len(path) for path in nx.all_simple_paths(G, node, node)]
                level = max(paths) if paths else 0
                levels[level].append(node)
            
            # Convert to list of batches
            batches = [nodes for _, nodes in sorted(levels.items())]
            logging.info(f"Identified {len(batches)} processing batches")
            for i, batch in enumerate(batches):
                logging.debug(f"Batch {i+1}: {batch}")
            
            return batches
            
        except nx.NetworkXUnfeasible:
            logging.warning("Circular dependencies detected, falling back to sequential processing")
            return [[node] for node in G.nodes()]

    def get_optimal_processing_order(self, activities):
        """Determine optimal processing order considering dependencies"""
        G = self.analyze_dependencies(activities)
        batches = self.get_processing_batches(G)
        
        # Flatten batches into final processing order
        processing_order = [activity for batch in batches for activity in batch]
        
        logging.info("Determined optimal processing order:")
        for i, activity in enumerate(processing_order, 1):
            logging.debug(f"{i}. {activity}")
        
        return processing_order