import React, { useState, useEffect } from 'react';
import {
  Paper,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import axios from 'axios';
import Evaluation from './Evaluation';

const DecisionTreeChart = ({ data }) => {
  // ... (existing code)

  const [decisionTree, setDecisionTree] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const requestData = {
        arrayData: data,
      };

      try {
        const response = await axios.post(
          `http://127.0.0.1:8000/api/v1/rulebased/`,
          requestData
        );
        if (response.data != null) {
          console.log(response)
          setDecisionTree(response.data);
        }
      } catch (error) {
        console.error('Error sending POST request:', error);
      }
    };

  
      fetchData();
    
  }, []);

  return (
    <Paper style={{ padding: '20px' }}>
      {/* ... (existing code) */}
      {decisionTree ? (
        <Evaluation treeData={decisionTree} />
      ) : (
        <p>No decision tree data available.</p>
      )}
    </Paper>
  );
};

export default DecisionTreeChart;
