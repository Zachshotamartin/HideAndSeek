import { mountExperiment } from './index.js';
import './style.css';
const experiment = mountExperiment(document.getElementById('app'));
if (import.meta.hot) import.meta.hot.dispose(() => experiment.dispose());
