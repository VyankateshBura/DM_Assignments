import React, { useState,useEffect } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import ScatterPlotChart from '../../ScatterPlot/ScatterPlotChart';

import ReactPaginate from 'react-paginate';

const Statistics = ({ data ,url,title}) => {
  const [RegData1, setRegData1] = useState(null);
  const [RegData2, setRegData2] = useState(null);
  const [RegData3, setRegData3] = useState(null);
  const [RegData4, setRegData4] = useState(null);
  useEffect(() => {
    const fetchData = async () => {
      const requestData = {
        arrayData: data,
      };
      const rData = {
        arrayData: data,
        k :3
      }
      try {
        const response1 = await axios.post(`http://127.0.0.1:8000/api/v1/regression/`, requestData);
        const response2 = await axios.post(`http://127.0.0.1:8000/api/v1/naivebayes/`, requestData);
        const response3 = await axios.post(`http://127.0.0.1:8000/api/v1/knn/`, rData);
        const response4 = await axios.post(`http://127.0.0.1:8000/api/v1/ann/`, requestData);
        console.log("Response from backend ",response4)
        setRegData1(response1.data);
        setRegData2(response2.data);
        setRegData3(response3.data);
        setRegData4(response4.data);
        
      } catch (error) {
        console.error('Error sending POST request:', error);
      }
    };

    if (data.length > 0) {
      fetchData();
    }
  }, [data]);

  const renderConfusionMatrix = (matrix) => (
    <table>
      <tbody>
        {matrix.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {row.map((cell, colIndex) => (
              <td key={colIndex}>{cell}&nbsp;&nbsp;</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );

  // console.log(RegData['Confusion Matrix'])
  return (
    <div style={{ width: '100%', height: 'fit-content' }}>

{RegData1!=null ? (
        <div>
          <h2>Logistic Regression :</h2>
          {/* Display the confusion matrix, accuracy, precision, and recall for each measure */}
         
          <div >
              <hr></hr>
              F1 Score : <strong>{RegData1['F1-Score']}</strong>&nbsp;&nbsp;&nbsp;
              Accuracy:<strong>{RegData1['Accuracy']}</strong>&nbsp;&nbsp;&nbsp;
              Precision:<strong>{RegData1['Precision']}</strong>&nbsp;&nbsp;&nbsp;
              Recall:<strong>{RegData1['Recall']}</strong>&nbsp;&nbsp;&nbsp;
                <h5>Confusion Matrix:</h5>
                <strong>{RegData1!=null?renderConfusionMatrix(RegData1['Confusion Matrix']):""}</strong>
              <div>
              </div>
            </div>


        </div>
      ) : (
        <p>Loading the data...</p>
      )}
<hr></hr>
{RegData2!=null ? (
        <div>
          <h2>Naive Bayes Classifier :</h2>
          {/* Display the confusion matrix, accuracy, precision, and recall for each measure */}
         
          <div >
          <div>
            <hr />
            Accuracy: <strong>{RegData2['Accuracy']}</strong>&nbsp;&nbsp;&nbsp;
            Precision: <strong>{RegData2['Precision']}</strong>&nbsp;&nbsp;&nbsp;
            Recall: <strong>{RegData2['Recall']}</strong>&nbsp;&nbsp;&nbsp;
            Recognition Rate: <strong>{RegData2['Recognition Rate']}</strong>&nbsp;&nbsp;&nbsp;
            Misclassification Rate: <strong>{RegData2['Misclassification Rate']}</strong>&nbsp;&nbsp;&nbsp;
            <h5>Confusion Matrix:</h5>
            <strong>{RegData2 != null ? renderConfusionMatrix(RegData2['Confusion Matrix']) : ""}</strong>
            <div>
              <h5>Sensitivity:</h5>
              <strong>{RegData2 != null ? RegData2['Sensitivity'].join(', ') : ""}</strong>
            </div>
            <div>
              <h5>Specificity:</h5>
              <strong>{RegData2 != null ? RegData2['Specificity'].join(', ') : ""}</strong>
            </div>
          </div>
            </div>
        </div>
      ) : (
        <p>Loading the data...</p>
      )}
<hr></hr>
{RegData3!=null ? (
        <div>
          <h2>K Nearest Neighbour :</h2>
          {/* Display the confusion matrix, accuracy, precision, and recall for each measure */}
         
          <div >
          <div>
            <hr />
            Accuracy: <strong>{RegData3['Accuracy']}</strong>&nbsp;&nbsp;&nbsp;
            Precision: <strong>{RegData3['Precision']}</strong>&nbsp;&nbsp;&nbsp;
            Recall: <strong>{RegData3['Recall']}</strong>&nbsp;&nbsp;&nbsp;
            Recognition Rate: <strong>{RegData3['Recognition Rate']}</strong>&nbsp;&nbsp;&nbsp;
            Misclassification Rate: <strong>{RegData3['Misclassification Rate']}</strong>&nbsp;&nbsp;&nbsp;
            <h5>Confusion Matrix:</h5>
            <strong>{RegData3 != null ? renderConfusionMatrix(RegData3['Confusion Matrix']) : ""}</strong>
            <div>
              <h5>Sensitivity:</h5>
              <strong>{RegData3 != null ? RegData3['Sensitivity'].join(', ') : ""}</strong>
            </div>
            <div>
              <h5>Specificity:</h5>
              <strong>{RegData2 != null ? RegData2['Specificity'].join(', ') : ""}</strong>
            </div>
          </div>
            </div>
        </div>
      ) : (
        <p>Loading the data...</p>
      )}
<hr></hr>

{RegData4!=null ? (
        <div>
          <h2>ANN Classifier :</h2>
          {/* Display the confusion matrix, accuracy, precision, and recall for each measure */}
         
          <div >
              <hr></hr>
              F1 Score : <strong>{RegData4['F1-Score']}</strong>&nbsp;&nbsp;&nbsp;
              Accuracy:<strong>{RegData4['Accuracy']}</strong>&nbsp;&nbsp;&nbsp;
              Precision:<strong>{RegData4['Precision']}</strong>&nbsp;&nbsp;&nbsp;
              Recall:<strong>{RegData4['Recall']}</strong>&nbsp;&nbsp;&nbsp;
                <h5>Confusion Matrix:</h5>
                <strong>{RegData4!=null?renderConfusionMatrix(RegData4['Confusion Matrix']):""}</strong>
              <div>
              </div>
            </div>


        </div>
      ) : (
        <p>Loading the data...</p>
      )}
    </div>
  );
};

export default Statistics;
