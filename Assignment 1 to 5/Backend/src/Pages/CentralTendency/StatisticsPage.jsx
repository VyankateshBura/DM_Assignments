import React, { useState } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import Papa from 'papaparse';

const StatisticsPage = ({ data }) => {
  const columnNames = Object.keys(data[0]);

  const getColumnData = (columnName) => data.map((row) => row[columnName]);

  const calculateMean = (columnName) => {
    const columnData = getColumnData(columnName);
    const sum = columnData.reduce((acc, val) => acc + parseFloat(val), 0);
    return sum / columnData.length;
  };

  const calculateMedian = (columnName) => {
    const columnData = getColumnData(columnName);
    const sortedData = columnData.slice().sort((a, b) => parseFloat(a) - parseFloat(b));
    const middle = Math.floor(sortedData.length / 2);
    if (sortedData.length % 2 === 0) {
      return (parseFloat(sortedData[middle - 1]) + parseFloat(sortedData[middle])) / 2;
    } else {
      return parseFloat(sortedData[middle]);
    }
  };

  const calculateMode = (columnName) => {
    const columnData = getColumnData(columnName);
    const frequency = {};
    let mode = null;
    let maxFrequency = 0;
    columnData.forEach((value) => {
      frequency[value] = (frequency[value] || 0) + 1;
      if (frequency[value] > maxFrequency) {
        mode = value;
        maxFrequency = frequency[value];
      }
    });
    return mode;
  };

  const calculateMidrange = (columnName) => {
    const columnData = getColumnData(columnName);
    const min = Math.min(...columnData.map((val) => parseFloat(val)));
    const max = Math.max(...columnData.map((val) => parseFloat(val)));
    return (min + max) / 2;
  };

  const calculateVariance = (columnName) => {
    const columnData = getColumnData(columnName);
    const mean = calculateMean(columnName);
    const sumOfSquares = columnData.reduce((acc, val) => acc + (parseFloat(val) - mean) ** 2, 0);
    return sumOfSquares / columnData.length;
  };

  const calculateStandardDeviation = (columnName) => {
    return Math.sqrt(calculateVariance(columnName));
  };

  return (
    <div>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Column Name</TableCell>
              <TableCell>Mean</TableCell>
              <TableCell>Median</TableCell>
              <TableCell>Mode</TableCell>
              <TableCell>Midrange</TableCell>
              <TableCell>Variance</TableCell>
              <TableCell>Standard Deviation</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {columnNames.map((columnName) => (
              <TableRow key={columnName}>
                <TableCell>{columnName}</TableCell>
                <TableCell>{calculateMean(columnName).toFixed(2)}</TableCell>
                <TableCell>{calculateMedian(columnName).toFixed(2)}</TableCell>
                <TableCell>{calculateMode(columnName)}</TableCell>
                <TableCell>{calculateMidrange(columnName).toFixed(2)}</TableCell>
                <TableCell>{calculateVariance(columnName).toFixed(2)}</TableCell>
                <TableCell>{calculateStandardDeviation(columnName).toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Additional visualizations */}
      <div style={{ marginTop: '40px' }}>
        {columnNames.map((columnName) => (
          <div key={columnName} style={{ marginBottom: '20px' }}>
            <h3>{columnName}</h3>
            <BarChart width={800} height={300} data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={columnName} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey={columnName} fill="#8884d8" name={columnName} />
            </BarChart>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StatisticsPage;
