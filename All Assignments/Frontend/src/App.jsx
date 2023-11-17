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
import BIRCH from './Pages/BIRCH/BIRCH';
import DBSCAN from './Pages/DBSCAN/DBSCAN';
import K_Means from './Pages/K-Means/K-Means';
import K_Medoids from './Pages/K-Medoids/K-Medoids';
import HierarchialDIANA from './Pages/Hierarchial/DIANA/HierarchialDIANA';
import HierarchialAGNES from './Pages/Hierarchial/AGNES/HierarchialAGNES';
import Apriori from './Pages/Apriori/Apriori';
import Crawler from "./Pages/Crawler/Crawler"
import PageRank from "./Pages/PageRank/PageRank"
import HITS from "./Pages/HITS/HITS"


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
            <Route path='/assignment6/section1' element={<Dashboard/>}/>
            <Route path='/assignment6/section2' element={<HierarchialAGNES/>}/>
            <Route path='/assignment6/section3' element={<K_Means/>}/>
            <Route path='/assignment6/section4' element={<K_Medoids/>}/>
            <Route path='/assignment6/section5' element={<BIRCH/>}/>
            <Route path='/assignment6/section6' element={<DBSCAN/>}/>
            <Route path='/assignment6/section7' element={<HierarchialDIANA/>}/>
            <Route path='/assignment7/section1' element={<Dashboard/>}/>
            <Route path='/assignment7/section2' element={<Apriori/>}/>
            <Route path='/assignment8/section1' element={<Crawler/>}/>
            <Route path='/assignment8/section2' element={<PageRank/>}/>
            <Route path='/assignment8/section3' element={<HITS/>}/>
          </Routes>
        </BrowserRouter>
      </FileData.Provider>
  )
}

export default App