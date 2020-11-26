import { Router } from "express";
import * as tf from "@tensorflow/tfjs-node";

const swaggerUiOptions = {
  customCss: ".swagger-ui .topbar { display: none }",
};

const router = Router();

const ports = [
  { name: "Paris" },
  { name: "Antwerp" },
  { name: "Brussels" },
  { name: "Ghent" },
  { name: "Hasselt" },
];

const vessels = [
  { name: "Maersk Variatie" },
  { name: "Maersk Sophie" },
  ,
  { name: "Maersk Moroc" },
  { name: "Maersk Algerie" },
  { name: "Maersk India" },
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

// Vessels have fixed routes between two ports, except in 10% of cases
const vesselXport = vessels.map((v) => {
  return {
    vessel: v,
    ...getRandomPortCombo(),
  };
});

console.log(vesselXport);

const datapoints = [];
for (let i = 0; i < 1000; i++) {
  // For each vessel, make x journeys, with a 10% chance of having a random journey

  vesselXport.forEach((ev) => {
    let portA = ev.portA;
    let portB = ev.portB;
    if (Math.random() < 0.1) {
      // random journey
      const ports = getRandomPortCombo();
      portA = ports.portA;
      portB = ports.portB;
    }

    console.log(
      ev.vessel.name + " will travel from " + portA.name + " to " + portB.name
    );
    datapoints.push({
      v: ev.vessel,
      portA: portA,
      portB: portB,
    });
  });
}

// Book routes
router.get("/", (req, res) => {
  const model = tf.sequential();
  // model.add(tf.layers.dense({ inputShape: [], units: 2, useBias: true }));
  res.end();
});

export default router;
