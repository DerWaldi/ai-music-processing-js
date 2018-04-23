import React from 'react';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import { Switch, Route, Link, BrowserRouter } from 'react-router-dom'

import AppBar from 'material-ui/AppBar';
import MenuItem from 'material-ui/MenuItem';
import Drawer from 'material-ui/Drawer';

import ChromagramTest from './examples/ChromagramTest';
import KerasTest from './examples/KerasTest';
import ChordRecognitionTest from './examples/ChordRecognitionTest';
 
class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
        open: true
    }
  }

  render() {
    return (
      <BrowserRouter>
        <MuiThemeProvider>  
          <Drawer open={this.state.open}>
            <Link className="MenuLink" to='/'><MenuItem>Experiment 1: Chromagram</MenuItem></Link>
            <Link className="MenuLink" to='/lstm'><MenuItem>Experiment 2: KerasJS</MenuItem></Link>
            <Link className="MenuLink" to='/chords'><MenuItem>Experiment 3: Chords</MenuItem></Link>
          </Drawer>
          <div style={{marginLeft: this.state.open ? '256px' : 0}}>
            <AppBar title="AI Music Processing JS" onLeftIconButtonClick={() => {this.setState({ open: !this.state.open });}}/>  
            <div style={{padding: 20}}>
              <Switch>
                <Route exact path='/' component={ChromagramTest}/>
                <Route path='/lstm' component={KerasTest}/>
                <Route path='/chords' component={ChordRecognitionTest}/>
              </Switch>
            </div>
          </div>
        </MuiThemeProvider>
      </BrowserRouter>
    );
  }
}

export default App;
