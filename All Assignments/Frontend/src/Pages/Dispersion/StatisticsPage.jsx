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

  const calculateRange = (columnName) => {
    const columnData = getColumnData(columnName);
    const min = Math.min(...columnData.map((val) => parseFloat(val)));
    const max = Math.max(...columnData.map((val) => parseFloat(val)));
    return max - min;
  };

  const calculateQuartiles = (columnName) => {
    const columnData = getColumnData(columnName);
    const sortedData = columnData.slice().sort((a, b) => parseFloat(a) - parseFloat(b));
    const n = sortedData.length;
  
    if (n === 0) {
      return { q1: 0, q2: 0, q3: 0 };
    }
  
    let q1, q2, q3;
  
    if (n % 2 === 0) {
      q2 = (parseFloat(sortedData[n / 2 - 1]) + parseFloat(sortedData[n / 2])) / 2;
      const lowerHalf = sortedData.slice(0, n / 2);
      const upperHalf = sortedData.slice(n / 2);
  
      if (lowerHalf.length % 2 === 0) {
        q1 = (parseFloat(lowerHalf[lowerHalf.length / 2 - 1]) + parseFloat(lowerHalf[lowerHalf.length / 2])) / 2;
        q3 = (parseFloat(upperHalf[upperHalf.length / 2 - 1]) + parseFloat(upperHalf[upperHalf.length / 2])) / 2;
      } else {
        q1 = parseFloat(lowerHalf[Math.floor(lowerHalf.length / 2)]);
        q3 = parseFloat(upperHalf[Math.floor(upperHalf.length / 2)]);
      }
    } else {
      q2 = parseFloat(sortedData[Math.floor(n / 2)]);
      const lowerHalf = sortedData.slice(0, Math.floor(n / 2));
      const upperHalf = sortedData.slice(Math.floor(n / 2) + 1);
  
      if (lowerHalf.length % 2 === 0) {
        q1 = (parseFloat(lowerHalf[lowerHalf.length / 2 - 1]) + parseFloat(lowerHalf[lowerHalf.length / 2])) / 2;
        q3 = (parseFloat(upperHalf[upperHalf.length / 2 - 1]) + parseFloat(upperHalf[upperHalf.length / 2])) / 2;
      } else {
        q1 = parseFloat(lowerHalf[Math.floor(lowerHalf.length / 2)]);
        q3 = parseFloat(upperHalf[Math.floor(upperHalf.length / 2)]);
      }
    }
  
    return { q1, q2, q3 };
  };
  

  const calculateInterquartileRange = (columnName) => {
    const { q1, q3 } = calculateQuartiles(columnName);
    return q3 - q1;
  };

  const calculateFiveNumberSummary = (columnName) => {
    const columnData = getColumnData(columnName);
    const sortedData = columnData.slice().sort((a, b) => parseFloat(a) - parseFloat(b));
    const n = sortedData.length;
    const q1 = sortedData[Math.floor(n / 4)];
    const q2 = sortedData[Math.floor(n / 2)];
    const q3 = sortedData[Math.floor((3 * n) / 4)];
    const min = sortedData[0];
    const max = sortedData[n - 1];
    return { min, q1, q2, q3, max };
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
              <TableCell>Range</TableCell>
              <TableCell>Q1</TableCell>
              <TableCell>Q2 (Median)</TableCell>
              <TableCell>Q3</TableCell>
              <TableCell>IQR</TableCell>
              <TableCell>Five Number Summary</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {columnNames.map((columnName) => (
              <TableRow key={columnName}>
                <TableCell>{columnName}</TableCell>
                <TableCell>{calculateMean(columnName)}</TableCell>
                <TableCell>{calculateMedian(columnName)}</TableCell>
                <TableCell>{calculateMode(columnName)}</TableCell>
                <TableCell>{calculateMidrange(columnName)}</TableCell>
                <TableCell>{calculateVariance(columnName)}</TableCell>
                <TableCell>{calculateStandardDeviation(columnName)}</TableCell>
                <TableCell>{calculateRange(columnName)}</TableCell>
                <TableCell>{calculateQuartiles(columnName).q1}</TableCell>
                <TableCell>{calculateQuartiles(columnName).q2}</TableCell>
                <TableCell>{calculateQuartiles(columnName).q3}</TableCell>
                <TableCell>{calculateInterquartileRange(columnName)}</TableCell>
                <TableCell>
                  {Object.values(calculateFiveNumberSummary(columnName)).map((value) => value).join(', ')}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
};

export default StatisticsPage;
