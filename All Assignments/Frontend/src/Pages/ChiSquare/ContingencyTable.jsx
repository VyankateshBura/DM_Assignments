import React from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

const ContingencyTable = ({ data, chi2, p, dof, relationshipStatus }) => {
  // Extract keys (column names) and values (row data)
  console.log("Data parameter :",data)
  const columnKeys = (data != null && data != undefined) ? Object.keys(data) : [];
  const rowKeys = (data != null && data != undefined && columnKeys.length > 0) ? Object.keys(data[columnKeys[0]]) : [];  
  // Object.keys(data[columnKeys[0]]);

  return (
    <Paper elevation={3} style={{ padding: '20px' }}>
      <h2>Contingency Table</h2>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell></TableCell> {/* Empty cell for the top-left corner */}
              {columnKeys.map((columnKey) => (
                <TableCell key={columnKey}>{columnKey}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rowKeys.map((rowKey) => (
              <TableRow key={rowKey}>
                <TableCell>{rowKey}</TableCell> {/* Row label */}
                {columnKeys.map((columnKey) => (
                  <TableCell key={columnKey}>{data[columnKey][rowKey]}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <h2>Chi-Square Test</h2>
      <p>Chi-Square Value: {chi2}</p>
      <p>P-value: {p}</p>
      <p>Degrees of Freedom: {dof}</p>
      <p>Relationship Status: {relationshipStatus}</p>
    </Paper>
  );
};

export default ContingencyTable;
