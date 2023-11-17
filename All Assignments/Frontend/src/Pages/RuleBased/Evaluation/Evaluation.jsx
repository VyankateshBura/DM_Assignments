import React, { useState, useEffect } from 'react';
import {Treebeard} from 'react-treebeard';
import DecisionTree1 from "../DecisionTree/DecisionTree1.png"
import DecisionTree2 from "../DecisionTree/DecisionTree2.png"
// import DecisionTree3 from "../DecisionTree/DecisionTree3.png"
const Evaluation = ({ treeData }) => {
  const [treeStructure, setTreeStructure] = useState(null);
  console.log("Tree path",treeData,Object.keys(treeData)[0],Object.keys(treeData)[1],Object.keys(treeData)[2])

  const [Trees,setTrees] = useState([])
 

  useEffect(() => {
    setTrees([DecisionTree1,DecisionTree2])
  }, []);

  const ExtractedRules = (rules) => {
    // Function to recursively create HTML elements for displaying rules
    console.log("The rules are :",rules)
    if (!rules) {
      // Handle the case when rules are undefined or null
      return <div>Loading...</div>; // You can replace this with your loading indicator
    }
    
    const renderRule = (rule, index) => (
      <div key={index} style={{ marginLeft: '20px' }}>
        <strong>Condition:</strong>
        <p> {rule.condition}</p>
        {Array.isArray(rule.outcome) ? (
          <ul>
            {rule.outcome.map((subRule, subIndex) => (
              <li key={subIndex}>{renderRule(subRule, subIndex)}</li>
            ))}
          </ul>
        ) : (
          <div>
            <strong>Outcome:</strong>
            {rule.outcome}
          </div>
          
        )}
      </div>
    );
  
    const formattedRules = [];

    // Loop through the rules array, grouping conditions and outcomes
    for (let i = 0; i < rules.length; i += 2) {
      const condition = rules[i];
      const outcome = rules[i + 1];

      formattedRules.push({ condition, outcome });
    }

    return (
      <div>
        <h1>Extracted Rules</h1>
        <div>
          {formattedRules.map((rule, index) => (
            <div key={index} className="rule">
              {renderRule(rule, index)}
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  // const renderConfusionMatrix = (matrix) => (
  //   <table>
  //     <tbody>
  //       {matrix.map((row, rowIndex) => (
  //         <tr key={rowIndex}>
  //           {row.map((cell, colIndex) => (
  //             <td key={colIndex}>{cell}</td>
  //           ))}
  //         </tr>
  //       ))}
  //     </tbody>
  //   </table>
  // );

  return (
    <div style={{ width: '100%', height: 'fit-content' }}>

{treeData ? (
        <div>
          <h2>Decision Tree Evaluation:</h2>
            <div>{ExtractedRules(JSON.parse(treeData.Tree))}</div>
 
              <hr></hr>
              Coverage: <strong>{treeData.Coverage}</strong>&nbsp;&nbsp;&nbsp;
              Accuracy:<strong>{treeData.Accuracy}</strong>&nbsp;&nbsp;&nbsp;
              Toughness:<strong>{treeData.Toughness}</strong>&nbsp;&nbsp;&nbsp;
              <div>
                
              </div>
              
            </div>

      ) : (
        <p>Loading tree data...</p>
      )}
    </div>
  );
};

export default Evaluation;
