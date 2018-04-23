import React from 'react';
import CircularProgress from 'material-ui/CircularProgress';
import TextField from 'material-ui/TextField';
import axios from 'axios';

import KerasJS from 'keras-js'

const MAXLEN = 200
// start index, out-of-vocabulary index
// see https://github.com/keras-team/keras/blob/master/keras/datasets/imdb.py
const START_WORD_INDEX = 1
const OOV_WORD_INDEX = 2
const INDEX_FROM = 3

const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.bin'

const ADDITIONAL_DATA_FILEPATHS_PROD = {
    wordIndex: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json',
    wordDict: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json',
    testSamples: 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_dataset_test.json'
  }
 
class KerasTest extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            inputText: "",
            loading: false,
            output: 0
        };
        this.model = new KerasJS.Model({
            filepath: MODEL_FILEPATH_PROD,
            gpu: false
        });

        this.loadAdditionalData();
        
        //this.model.events.on('loadingProgress', this.handleLoadingProgress)
        //this.model.events.on('initProgress', this.handleInitProgress)
    }

    loadAdditionalData() {
        this.modelLoading = true
        const reqs = ['wordIndex', 'wordDict', 'testSamples'].map(key => {
            return axios.get(ADDITIONAL_DATA_FILEPATHS_PROD[key])
        });
        axios.all(reqs).then(
            axios.spread((wordIndex, wordDict, testSamples) => {
                this.wordIndex = wordIndex.data
                this.wordDict = wordDict.data
                this.testSamples = testSamples.data
                console.log("loadAdditionalData completed!");
            })
        );
    }

    componentDidMount() {
        this.setState({loading: true});
        this.model.ready().then(() => {
            console.log("model loading completed!");
            this.setState({loading: false});
        });
    }
    inputChanged(newValue) {
        this.inputTextParsed = newValue
            .trim()
            .toLowerCase()
            .split(/[\s.,!?]+/gi)
        this.input = new Float32Array(MAXLEN)
        // by convention, use 2 as OOV word
        // reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
        // see https://github.com/keras-team/keras/blob/master/keras/datasets/imdb.py
        let indices = this.inputTextParsed.map(word => {
            const index = this.wordIndex[word]
            return !index ? OOV_WORD_INDEX : index + INDEX_FROM
        })
        indices = [START_WORD_INDEX].concat(indices)
        indices = indices.slice(-MAXLEN)
        // padding and truncation (both pre sequence)
        const start = Math.max(0, MAXLEN - indices.length)
        for (let i = start; i < MAXLEN; i++) {
            this.input[i] = indices[i - start]
        }
        this.model.predict({ input: this.input }).then(outputData => {
            this.setState({output: new Float32Array(outputData.output)[0]});
        });
    }
    render () {
        return(
            <div>
                <h3>KerasJS IMDB RNN Test</h3>
                <p>Enter a Text and it will be estimate whether it is positive (--> 1) or bad (--> 0)</p>
                <TextField 
                    value={this.state.inputText} 
                    onChange={(e, newValue) => {this.setState({inputText: newValue}); this.inputChanged(newValue);}} 
                    hintText="Enter a Text"
                    multiLine={true}
                /><br />
                <p>Score: {this.state.output.toFixed(2)}</p>
                {this.state.loading > 0 ?
                <CircularProgress 
                    style={{position: 'fixed', left: 'calc(50% - 40px)', top: 'calc(50% - 40px)', zIndex: 1000}}
                    size={80}
                    thickness={7}
                /> : null}
            </div>
        );
    }
}
 
export default KerasTest;