import React,{useState} from 'react'
import {BrowserRouter,Routes,Route} from 'react-router-dom';
import Dashboard from './Pages/dashboard/Dashboard';
import Home from './Pages/Home';
import SignIn from './Pages/sign-in/SignIn';
import SignUp from './Pages/sign-up/SignUp';
import {getData} from "./Pages/Data"
import CentralTendency from "./Pages/CentralTendency/CentralTendency"
import Histogram from './Pages/Histogram/Histogram';
import ScatterPlot from './Pages/ScatterPlot/ScatterPlot';
import Dispersion from './Pages/Dispersion/Dispersion';
import BoxPlot from "./Pages/BoxPlot/BoxPlot"
import QuantilePlot from './Pages/QuantilePlot/QuantilePlot';
import ChiSquare from "./Pages/ChiSquare/ChiSquare"
import MinMax from './Pages/Normalization/MinMax/MinMax';
import ZScore from './Pages/Normalization/Z-score/ZScore';
import DecimalScaling from './Pages/Normalization/DecimalScaling/DecimalScaling';
import Correlation from './Pages/Correlation/Correlation';
import DecisionTree from './Pages/RuleBased/DecisionTree/DecisionTree';
import Evaluate from './Pages/RuleBased/Evaluation/Evaluate';
import Regression from './Pages/Classifiers/Regression/Regression';



export const FileData = React.createContext();
const App = () => {
  const [fdata, setFdata] = useState(null)
 
  
  return (
      <FileData.Provider value={{fdata,setFdata}}>
        <BrowserRouter>
          <Routes>
            <Route path='/' element={<Home/>}/>
            <Route path='/signin' element={<SignIn/>}/>
            <Route path='/signup' element={<SignUp/>}/>
            <Route path='/assignment1/section1' element={<Dashboard/>}/>
            <Route path='/assignment1/section2' element={<CentralTendency/>}/>
            <Route path='/assignment1/section3' element={<Histogram/>}/>
            <Route path='/assignment1/section4' element={<Dispersion/>}/>
            <Route path='/assignment1/section5' element={<ScatterPlot/>}/>
            <Route path='/assignment1/section6' element={<BoxPlot/>}/>
            <Route path='/assignment1/section7' element={<QuantilePlot/>}/>
            <Route path='/assignment2/section1' element={<Dashboard/>}/>
            <Route path='/assignment2/section2' element={<ChiSquare/>}/>
            <Route path='/assignment2/section3' element={<MinMax/>}/>
            <Route path='/assignment2/section4' element={<DecimalScaling/>}/>
            <Route path='/assignment2/section5' element={<ZScore/>}/>
            <Route path='/assignment2/section6' element={<Correlation/>}/>
            <Route path='/assignment3/section1' element={<Dashboard/>}/>
            <Route path='/assignment3/section2' element={<DecisionTree/>}/>
            <Route path='/assignment4/section1' element={<Dashboard/>}/>
            <Route path='/assignment4/section2' element={<Evaluate/>}/>
            <Route path='/assignment5/section1' element={<Dashboard/>}/>
            <Route path='/assignment5/section2' element={<Regression/>}/>
          </Routes>
        </BrowserRouter>
      </FileData.Provider>
  )
}

export default App