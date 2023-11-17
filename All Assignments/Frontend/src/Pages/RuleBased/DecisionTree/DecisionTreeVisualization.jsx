import React, { useState, useEffect } from 'react';
import {Treebeard} from 'react-treebeard';
import DecisionTree1 from "./DecisionTree1.png"
import DecisionTree2 from "./DecisionTree2.png"
// import DecisionTree3 from "./DecisionTree3.png"
const DecisionTreeVisualization = ({ treeData }) => {
  const [treeStructure, setTreeStructure] = useState(null);
  console.log("Tree path",treeData,Object.keys(treeData)[0],Object.keys(treeData)[1],Object.keys(treeData)[2])

  const [Trees,setTrees] = useState([])
 

  useEffect(() => {
    setTrees([DecisionTree1,DecisionTree2])
  }, []);

  
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

  return (
    <div style={{ width: '100%', height: 'fit-content' }}>

{treeData ? (
        <div>
          <h2>Decision Tree Diagram:</h2>
          {/* Display the confusion matrix, accuracy, precision, and recall for each measure */}
          {Object.keys(treeData).map((measureValue,index) => (
            <div key={index}>
              <hr></hr>
              Measure: <strong>{treeData[measureValue].measure}</strong>&nbsp;&nbsp;&nbsp;
              Accuracy:<strong>{treeData[measureValue].accuracy}</strong>&nbsp;&nbsp;&nbsp;
              Precision:<strong>{treeData[measureValue].precision}</strong>&nbsp;&nbsp;&nbsp;
              Recall:<strong>{treeData[measureValue].recall}</strong>&nbsp;&nbsp;&nbsp;
                <h5>Confusion Matrix:</h5>
                <strong>{renderConfusionMatrix(treeData[measureValue].confusion_matrix)}</strong>
                <h5>Tree Image:</h5>
                <img src={Trees[index]} height={500} />
              <div>
                
              </div>
              
            </div>
          ))}
        </div>
      ) : (
        <p>Loading tree data...</p>
      )}
    </div>
  );
};

export default DecisionTreeVisualization;
