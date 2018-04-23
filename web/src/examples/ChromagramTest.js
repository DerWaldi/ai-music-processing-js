import React from 'react';
import RaisedButton from 'material-ui/RaisedButton';
import Meyda from 'meyda';
import Plot from 'react-plotly.js';
import CircularProgress from 'material-ui/CircularProgress';
 
class ChromagramTest extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            rmsFeatures: null,
            chromaFeatures: null,
            processing: false
        };
    }
    onFileChange() {
        if (this.fileInput.files.length > 0) {
            var file = this.fileInput.files[0];
            // URL.createObjectURL(this.fileInput.files[0])
            var context = new (window.AudioContext || window.webkitAudioContext)();
            var fileReader = new FileReader();
        
            var hopSize = 512;
            var frameSize = 2048;

            fileReader.onload = (fileEvent) => {
                var data = fileEvent.target.result;
                this.setState({processing: true});  
                context.decodeAudioData(data, (buffer) => {
                    var channelBuffer = buffer.getChannelData(0);

                    var chromaFeatures = [[], [], [], [], [], [], [], [], [], [], [], []];
                    var rmsFeatures = [];

                    for(var i = 0; i + frameSize < channelBuffer.length; i += hopSize) {
                        var features = Meyda.extract(['chroma', 'rms'], channelBuffer.slice(i, i + frameSize));
                        rmsFeatures.push(features.rms);
                        for(var j = 0; j < features.chroma.length; j++) {
                            chromaFeatures[j].push(features.chroma[j]);
                        }     
                    } 

                    this.setState({processing: false, chromaFeatures: chromaFeatures, rmsFeatures: rmsFeatures});                   
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
                {this.state.chromaFeatures != null ?
                <div>
                    <Plot
                        data={[{
                            z: this.state.chromaFeatures,
                            type: 'heatmap'
                        }]}
                        layout={ {width: 1600, height: 400, title: 'Chromagram'} }
                    /><br/><br/>
                    <Plot
                        data={[{
                            y: this.state.rmsFeatures,
                            type: 'lines'
                        }]}
                        layout={ {width: 1600, height: 400, title: 'RMS'} }
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
 
export default ChromagramTest;