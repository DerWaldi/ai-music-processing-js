import React from 'react';
import RaisedButton from 'material-ui/RaisedButton';
import Meyda from 'meyda';
import Plot from 'react-plotly.js';
import CircularProgress from 'material-ui/CircularProgress';

//import KerasJS from 'keras-js'        
import * as tf from '@tensorflow/tfjs';

var math = require('mathjs');

const MODEL_FILEPATH_PROD = 'tfjs_chords/model.json'
//const MODEL_FILEPATH_PROD = 'https://transcranial.github.io/keras-js-demos-data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.bin'
 
class ChordRecognitionTest extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            chordPath: null,
            processing: false
        };
    }

    async componentDidMount() {
        this.setState({processing: true});
        this.model = await tf.loadModel(MODEL_FILEPATH_PROD);
        this.setState({processing: false});
    }

    /*

    componentDidMount() {
        this.setState({processing: true});
        tf.loadModel(MODEL_FILEPATH_PROD).then(model => {
            this.model = model;
            this.setState({processing: false});
        });
    }
    */

    cut_sequences(a, max_seq_size) {
        var n = a.length;
        var n_cut = a.length - a.length % max_seq_size

        var result = [];
        for(var i = 0; i < n_cut - max_seq_size; i+=max_seq_size) {
            result.push(a.slice(i, i + max_seq_size));
        }
        return math.reshape(result, [result.length, max_seq_size, 12, 1]);
    }

    analyzeChords(chromaFeatures) {
        const max_seq_size = 32;
        var predictions;
        tf.tidy(() => {
            var u = this.cut_sequences(chromaFeatures, max_seq_size)
            const b = tf.tensor(u, [u.length, max_seq_size, 12, 1]);       
            const result = this.model.predict(b);
            const predictions_tensor = result.reshape([u.length*max_seq_size,25]).argMax(1);
            predictions_tensor.print();
            predictions = Array.from(predictions_tensor.dataSync());
        });
        return predictions;
    }

    onFileChange() {
        if (this.fileInput.files.length > 0) {
            var file = this.fileInput.files[0];
            // URL.createObjectURL(this.fileInput.files[0])
            var context = new (window.AudioContext || window.webkitAudioContext)();
            var fileReader = new FileReader();
        
            var hopSize = 2048;
            var frameSize = 4096;

            fileReader.onload = (fileEvent) => {
                var data = fileEvent.target.result;
                this.setState({processing: true});  
                context.decodeAudioData(data, (buffer) => {
                    var channelBuffer = buffer.getChannelData(0);

                    var chromaFeatures = [];
                    var rmsFeatures = [];

                    for(var i = 0; i + frameSize < channelBuffer.length; i += hopSize) {
                        var features = Meyda.extract(['chroma'], channelBuffer.slice(i, i + frameSize));
                        chromaFeatures.push(features.chroma); 
                    } 

                    var chords = this.analyzeChords(chromaFeatures);
                    
                    console.log(chords);

                    this.setState({processing: false, chordPath: chords});                   
                }, (e) => {
                    console.error('There was an error decoding ' + file.name);
                });
            };        
            fileReader.readAsArrayBuffer(file);
        }
    }
    render () {
        return(
            <div>
                <h3>Chromagram and RMS Features form MP3-Files</h3>
                <RaisedButton label="Upload MP3 File" onClick={() => {this.fileInput.click()}} /><br/><br/>
                <input type="file" ref={c => this.fileInput = c } onChange={() => {this.onFileChange()}} hidden/>
                {this.state.chordPath != null ?
                <div>
                    <Plot
                        data={[{
                            y: this.state.chordPath,
                            type: 'lines'
                        }]}
                        layout={ {width: 1600, height: 400, title: 'Chord Path'} }
                    />
                </div> : null}
                {this.state.processing > 0 ?
                <CircularProgress 
                    style={{position: 'fixed', left: 'calc(50% - 40px)', top: 'calc(50% - 40px)', zIndex: 1000}}
                    size={80}
                    thickness={7}
                /> : null}
            </div>
        );
    }
}
 
export default ChordRecognitionTest;