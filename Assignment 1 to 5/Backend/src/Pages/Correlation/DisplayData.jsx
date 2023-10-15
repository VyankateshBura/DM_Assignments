import React from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

const DisplayData = ({covariance,PC, relationshipStatus }) => {
  // Extract keys (column names) and values (row data)
  // console.log(covariance,PC,relationshipStatus)
  return (
    <Paper elevation={3} style={{ padding: '20px' }}>
      <h2>Correlation Analysis</h2>
      <p>Pearson correlation coefficient: {covariance}</p>
      <p>Covariance: {PC}</p>
      <p>Relationship Status: {relationshipStatus}</p>
    </Paper>
  );
};

export default DisplayData;
