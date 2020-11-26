import e, { Router } from "express";
import * as tf from "@tensorflow/tfjs-node";
import { data } from "@tensorflow/tfjs-node";

const swaggerUiOptions = {
  customCss: ".swagger-ui .topbar { display: none }",
};

const router = Router();

type Port = {
  name:string;
  x:number;
  y:number
}
type Vessel = {
  name:string;
  index:number
}
const ports:Array<Port> = [
  { name: "Paris",x:48.864716,y:2.349014},
  { name: "Antwerp",x:51.220504,y:4.473948},
  { name: "Brussels" ,x:50.850346,y:4.351721 },
  { name: "Ghent" ,x:51.054340,y:3.717424},
  { name: "Hasselt",x:50.93106,y:5.33781},
];

const vessels :Array<Vessel>= [
  { name: "Maersk Variatie",index:0 },
  { name: "Maersk Sophie",index:1 },
  { name: "Maersk Moroc",index:2 },
  { name: "Maersk Algerie" ,index:3},
  { name: "Maersk India",index:4},
];

function getRandomPortCombo() {
  const portAindex = Math.floor(Math.random() * ports.length);
  let portBIndex = Math.floor(Math.random() * ports.length);
  while (portBIndex == portAindex) {
    portBIndex = Math.floor(Math.random() * ports.length);
  }
  return {
    portA: ports[portAindex],
    portB: ports[portBIndex],
  };
}


function normalize(value:any, min:any, max:any) {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value - min) / (max - min);
}
function denormalize(value:any, min:any, max:any) {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value * (max - min))+min;
}
const maxCoordinateX = Math.max(...ports.map(e=>e.x));
const maxCoordinateY = Math.max(...ports.map(e=>e.y));
const maxIndexVessel = Math.max(...vessels.map(e=>e.index));
const minCoordinateX = Math.min(...ports.map(e=>e.x));
const minCoordinateY = Math.min(...ports.map(e=>e.y));
const minIndexVessel = Math.min(...vessels.map(e=>e.index));

const vesselXport = vessels.map((v) => {
  return {
    vessel: v,
    ...getRandomPortCombo(),
  };
});

console.log(vesselXport);

const datapoints:Array<{
  vessel:Vessel;
  portA:Port,
  portB:Port,
}> = [];
for (let i = 0; i < 1000; i++) {
  // For each vessel, make x journeys, with a x% chance of having a random journey

  vesselXport.forEach((ev) => {
    let portA = ev.portA;
    let portB = ev.portB;
    if (Math.random() < 0.02) {
      // random journey
      const ports = getRandomPortCombo();
      portB = ports.portB;
    }


    datapoints.push({
      vessel: ev.vessel,
      portA: portA,
      portB: portB,
    });
  });
}

function convertToTensor(data:Array<{
  vessel:Vessel,
  portA:Port,
  portB:Port,
}>) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor

    const inputs = data.map(e=>Uint8Array.from([normalize(e.vessel.index,minIndexVessel,maxIndexVessel),normalize(e.portA.x,minCoordinateX,maxCoordinateX),normalize(e.portA.y,minCoordinateY,maxCoordinateY)] ))
    const labels = data.map(e => [normalize(e.portB.x,minCoordinateX,maxCoordinateX),normalize(e.portB.y,minCoordinateY,maxCoordinateY)]);

    const inputTensor = tf.tensor2d(inputs);
    const labelTensor = tf.tensor2d(labels);

    return {
      inputs: inputTensor,
      labels: labelTensor,
    }
  });  
}

async function trainModel(model:tf.Sequential, inputs:any, labels:any) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });
  
  const batchSize = 50;
  const epochs = 20;
  
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
  });
}



function testModel(model:tf.Sequential, input:tf.Tensor<tf.Rank>, normalizationData:any) {
  const {labelMax} = normalizationData;  
  
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const preds= tf.tidy(() => {
  
  
    const preds = model.predict(input) as tf.Tensor;      
    


    // Un-normalize the data
    return preds.dataSync();
  });
  

  return preds;

  
  
  // tfvis.render.scatterplot(
  //   {name: 'Model Predictions vs Original Data'}, 
  //   {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
  //   {
  //     xLabel: 'Horsepower',
  //     yLabel: 'MPG',
  //     height: 300
  //   }
  // );
}

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [3],units:3,useBias:true}));
model.add(tf.layers.dense({units: 200}));
model.add(tf.layers.dense({units: 2}));



const tensorData = convertToTensor(datapoints);
 const {inputs, labels} = tensorData;
 
// Train the model  
trainModel(model, inputs, labels);

function distanceinKilometers(point1: {x:number,y:number}, point2: {x:number,y:number}): number {
  const p = 0.017453292519943295; // Math.PI / 180
  const c = Math.cos;
  const a =
      0.5 -
      c((point2.x - point1.x) * p) / 2 +
      (c(point1.x * p) *
          c(point2.x * p) *
          (1 - c((point2.y - point1.y) * p))) /
          2;

  return 12742 * Math.asin(Math.sqrt(a)); // 2 * R; R = 6371 km
}



// Book routes
router.get("/", async (req, res) => {

  const vessellll = vesselXport[Math.floor(Math.random()*vessels.length)];
  console.log(vessellll);
  const tensorInput = tf.tensor2d([normalize(vessellll.vessel.index,minIndexVessel,maxIndexVessel),normalize(vessellll.portA.x,minCoordinateX,maxCoordinateX),normalize(vessellll.portA.y,minCoordinateY,maxCoordinateY)],[1,3]);
  const index = testModel(model,tensorInput,tensorData)
  const xy = {x:denormalize(index[0],minCoordinateX,maxCoordinateX),y:denormalize(index[1],minCoordinateY,maxCoordinateY)};
  console.log(xy);
  const mini = ports.filter(p=>p.name != vessellll.portA.name).reduce((min,port)=>{
    const cost = distanceinKilometers(port,xy);
    if(cost < min.cost){
      min.cost = cost;
      min.port = port;
    }
    return min;
  },{port:ports[0],cost:distanceinKilometers(ports[0],xy)});
  //console.log(datapoints.filter(e=>e.vessel == vessellll.vessel))
  console.log(mini);


  res.end();
});

export default router;
