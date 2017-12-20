(function() {
  var ActivationFunction, App, NeuralNetwork, Neuron, NeuronLayer;

  ActivationFunction = {
    calculate: function(value) {
      return 1 / (1 + Math.exp(-value));
    },
    derivative: function(value) {
      return value * (1 - value);
    }
  };

  Neuron = (function() {
    Neuron.prototype.LEARNING_RATE = 0.5;

    function Neuron(bias) {
      this.bias = bias;
      this.weights = [];
    }

    Neuron.prototype.calculateWeightedSum = function() {
      var i, input, total, _i, _len, _ref;
      total = 0;
      _ref = this.inputs;
      for (i = _i = 0, _len = _ref.length; _i < _len; i = ++_i) {
        input = _ref[i];
        total += input * this.weights[i];
      }
      return total + this.bias;
    };

    Neuron.prototype.calculateOutput = function(inputs) {
      var output, weightedSum;
      this.inputs = inputs;
      weightedSum = this.calculateWeightedSum();
      output = ActivationFunction.calculate(weightedSum);
      this.lastOutput = output;
      return output;
    };

    Neuron.prototype.updateWeights = function() {
      var pdErrorWrtWeight, pdNetWrtInput, pdOutputWrtNet, w, _i, _ref, _results;
      pdOutputWrtNet = ActivationFunction.derivative(this.lastOutput);
      this.pdErrorWrtNet = this.pdErrorWrtOutput * pdOutputWrtNet;
      _results = [];
      for (w = _i = 0, _ref = this.weights.length - 1; 0 <= _ref ? _i <= _ref : _i >= _ref; w = 0 <= _ref ? ++_i : --_i) {
        pdNetWrtInput = this.inputs[w];
        pdErrorWrtWeight = this.pdErrorWrtNet * pdNetWrtInput;
        _results.push(this.weights[w] += this.LEARNING_RATE * pdErrorWrtWeight);
      }
      return _results;
    };

    return Neuron;

  })();

  NeuronLayer = (function() {
    function NeuronLayer(quantity) {
      var bias, i, _i, _ref;
      bias = Math.random();
      this.neurons = [];
      for (i = _i = 0, _ref = quantity - 1; 0 <= _ref ? _i <= _ref : _i >= _ref; i = 0 <= _ref ? ++_i : --_i) {
        this.neurons.push(new Neuron(bias));
      }
    }

    NeuronLayer.prototype.feedForward = function(inputs) {
      var neuron, outputs, _i, _len, _ref;
      outputs = [];
      _ref = this.neurons;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        neuron = _ref[_i];
        outputs.push(neuron.calculateOutput(inputs));
      }
      return outputs;
    };

    return NeuronLayer;

  })();

  NeuralNetwork = (function() {
    function NeuralNetwork(numInputs, numHiddenNeurons, numOutputNeurons) {
      var h, hiddenNeuron, input, outputNeuron, _i, _j, _k, _l, _len, _len1, _len2, _ref, _ref1, _ref2, _ref3;
      this.numInputs = numInputs;
      this.hiddenLayer = new NeuronLayer(numHiddenNeurons);
      this.outputLayer = new NeuronLayer(numOutputNeurons);
      _ref = this.hiddenLayer.neurons;
      for (h = _i = 0, _len = _ref.length; _i < _len; h = ++_i) {
        hiddenNeuron = _ref[h];
        for (input = _j = 0, _ref1 = numInputs - 1; 0 <= _ref1 ? _j <= _ref1 : _j >= _ref1; input = 0 <= _ref1 ? ++_j : --_j) {
          hiddenNeuron.weights.push(Math.random());
        }
      }
      _ref2 = this.outputLayer.neurons;
      for (_k = 0, _len1 = _ref2.length; _k < _len1; _k++) {
        outputNeuron = _ref2[_k];
        _ref3 = this.hiddenLayer.neurons;
        for (_l = 0, _len2 = _ref3.length; _l < _len2; _l++) {
          hiddenNeuron = _ref3[_l];
          outputNeuron.weights.push(Math.random());
        }
      }
    }

    NeuralNetwork.prototype.feedForward = function(inputs) {
      var hiddenOutputs;
      hiddenOutputs = this.hiddenLayer.feedForward(inputs);
      return this.outputLayer.feedForward(hiddenOutputs);
    };

    NeuralNetwork.prototype.getAverageError = function(trainingSet) {
      var data, error, o, outputResult, outputResults, setError, total, _i, _j, _len, _len1;
      total = 0;
      for (_i = 0, _len = trainingSet.length; _i < _len; _i++) {
        data = trainingSet[_i];
        setError = 0;
        outputResults = this.feedForward(data.inputs);
        for (o = _j = 0, _len1 = outputResults.length; _j < _len1; o = ++_j) {
          outputResult = outputResults[o];
          error = Math.abs(data.outputs[o] - outputResult);
          setError += error;
        }
        total += setError / data.outputs.length;
      }
      return total / trainingSet.length;
    };

    NeuralNetwork.prototype.train = function(trainingDataInputs, trainingDataOutputs) {
      var h, hiddenNeuron, o, outputNeuron, predictedOutput, targetOutput, _i, _j, _k, _len, _len1, _len2, _ref, _ref1, _ref2, _results;
      this.feedForward(trainingDataInputs);
      _ref = this.outputLayer.neurons;
      for (o = _i = 0, _len = _ref.length; _i < _len; o = ++_i) {
        outputNeuron = _ref[o];
        targetOutput = trainingDataOutputs[o];
        predictedOutput = outputNeuron.lastOutput;
        outputNeuron.pdErrorWrtOutput = targetOutput - predictedOutput;
        outputNeuron.updateWeights();
      }
      _ref1 = this.hiddenLayer.neurons;
      _results = [];
      for (h = _j = 0, _len1 = _ref1.length; _j < _len1; h = ++_j) {
        hiddenNeuron = _ref1[h];
        hiddenNeuron.pdErrorWrtOutput = 0;
        _ref2 = this.outputLayer.neurons;
        for (_k = 0, _len2 = _ref2.length; _k < _len2; _k++) {
          outputNeuron = _ref2[_k];
          hiddenNeuron.pdErrorWrtOutput += outputNeuron.weights[h] * outputNeuron.pdErrorWrtNet;
        }
        _results.push(hiddenNeuron.updateWeights());
      }
      return _results;
    };

    NeuralNetwork.prototype.render = function() {};

    return NeuralNetwork;

  })();

  App = (function() {
    App.prototype.NUM_HIDDEN_NEURONS = 5;

    App.prototype.NEURON_RADIUS = 30;

    App.prototype.NEURON_DIAMETER = 60;

    App.prototype.MAX_EDGE_WIDTH = 20;

    App.prototype.AVG_ERROR_HISTORY_LENGTH = 50;

    App.prototype.X_OFFSET = 100;

    function App() {
      var i, _i, _ref;
      this.neuralNetwork = new NeuralNetwork(2, this.NUM_HIDDEN_NEURONS, 1);
      this.$canvas = $('canvas');
      this.canvas = this.$canvas.get(0);
      this.ctx = this.canvas.getContext('2d');
      this.canvasWidth = this.$canvas.width();
      this.canvasHeight = this.$canvas.height();
      this.avgErrorHistory = [];
      for (i = _i = 1, _ref = this.AVG_ERROR_HISTORY_LENGTH; 1 <= _ref ? _i <= _ref : _i >= _ref; i = 1 <= _ref ? ++_i : --_i) {
        this.avgErrorHistory.push(0);
      }
    }

    App.prototype.clearCanvas = function() {
      return this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
    };

    App.prototype.renderNeuron = function(centerX, centerY) {
      this.ctx.beginPath();
      this.ctx.arc(centerX + this.X_OFFSET, centerY, this.NEURON_RADIUS, 0, 2 * Math.PI, false);
      this.ctx.fillStyle = 'red';
      this.ctx.fill();
      this.ctx.lineWidth = 1;
      this.ctx.strokeStyle = 'black';
      return this.ctx.stroke();
    };

    App.prototype.getNeuronSeparation = function(quantity) {
      return (this.canvasHeight - quantity * this.NEURON_DIAMETER) / (quantity + 1);
    };

    App.prototype.getNeuronCenterY = function(separation, n) {
      return separation * (n + 1) + this.NEURON_DIAMETER * n + this.NEURON_RADIUS;
    };

    App.prototype.renderNeuronLayer = function(leftX, quantity) {
      var i, separation, _i, _ref, _results;
      separation = this.getNeuronSeparation(quantity);
      _results = [];
      for (i = _i = 0, _ref = quantity - 1; 0 <= _ref ? _i <= _ref : _i >= _ref; i = 0 <= _ref ? ++_i : --_i) {
        _results.push(this.renderNeuron(leftX, this.getNeuronCenterY(separation, i)));
      }
      return _results;
    };

    App.prototype.renderEdge = function(x1, y1, x2, y2, weight) {
      var absMaxWeight, color, weights, width;
      weights = this.getWeights();
      absMaxWeight = _.max([Math.abs(_.min(weights)), Math.abs(_.max(weights))]);
      width = Math.ceil((Math.abs(weight) / absMaxWeight) * this.MAX_EDGE_WIDTH);
      if (weight < 0) {
        color = '#aaa';
      } else {
        color = 'black';
      }
      this.ctx.beginPath();
      this.ctx.moveTo(x1 + this.X_OFFSET, y1);
      this.ctx.lineTo(x2 + this.X_OFFSET, y2);
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = width;
      return this.ctx.stroke();
    };

    App.prototype.getWeights = function() {
      var neuron, neurons, weights, _i, _len;
      neurons = this.neuralNetwork.hiddenLayer.neurons.concat(this.neuralNetwork.outputLayer.neurons);
      weights = [];
      for (_i = 0, _len = neurons.length; _i < _len; _i++) {
        neuron = neurons[_i];
        weights = weights.concat(neuron.weights);
      }
      return weights;
    };

    App.prototype.renderLayerConnections = function(layer1X, layer1NeuronCount, layer2X, layer2Neurons) {
      var i, layer1Separation, layer2Separation, n, neuron, _i, _ref, _results;
      _results = [];
      for (i = _i = 0, _ref = layer1NeuronCount - 1; 0 <= _ref ? _i <= _ref : _i >= _ref; i = 0 <= _ref ? ++_i : --_i) {
        layer1Separation = this.getNeuronSeparation(layer1NeuronCount);
        _results.push((function() {
          var _j, _len, _results1;
          _results1 = [];
          for (n = _j = 0, _len = layer2Neurons.length; _j < _len; n = ++_j) {
            neuron = layer2Neurons[n];
            layer2Separation = this.getNeuronSeparation(layer2Neurons.length);
            _results1.push(this.renderEdge(layer1X, this.getNeuronCenterY(layer1Separation, i), layer2X, this.getNeuronCenterY(layer2Separation, n), neuron.weights[i]));
          }
          return _results1;
        }).call(this));
      }
      return _results;
    };

    App.prototype.render = function() {
      this.clearCanvas();
      this.renderLayerConnections(this.canvasWidth / 4, this.neuralNetwork.numInputs, this.canvasWidth / 2, this.neuralNetwork.hiddenLayer.neurons);
      this.renderLayerConnections(this.canvasWidth / 2, this.neuralNetwork.hiddenLayer.neurons.length, this.canvasWidth / 4 * 3, this.neuralNetwork.outputLayer.neurons);
      this.renderNeuronLayer(this.canvasWidth / 4, this.neuralNetwork.numInputs);
      this.renderNeuronLayer(this.canvasWidth / 2, this.neuralNetwork.hiddenLayer.neurons.length);
      return this.renderNeuronLayer(this.canvasWidth / 4 * 3, this.neuralNetwork.outputLayer.neurons.length);
    };

    App.prototype.updateSparkline = function() {
      var avgError;
      $('span#epoch').text(this.epoch.toLocaleString());
      avgError = _.num.round(this.neuralNetwork.getAverageError(this.trainingData), 5);
      $('span#avg_error').text(avgError);
      this.avgErrorHistory.push(avgError);
      this.avgErrorHistory = this.avgErrorHistory.slice(-this.AVG_ERROR_HISTORY_LENGTH);
      return $('div#avg_error_sparkline').sparkline(this.avgErrorHistory, {
        height: 50,
        width: 300,
        lineColor: '#FF5025',
        fillColor: '#f3f3f3',
        lineWidth: 2,
        chartRangeMin: 0,
        chartRangeMax: 0.3,
        spotColor: false,
        minSpotColor: false,
        maxSpotColor: false,
        highlightSpotColor: 'red'
      });
    };

    App.prototype.updatePredictionDisplay = function() {
      var $predictionContainer, data, joinedInputs, prediction, _i, _len, _ref, _results;
      $predictionContainer = $('div#status table');
      _ref = this.trainingData;
      _results = [];
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        data = _ref[_i];
        joinedInputs = data.inputs.join('');
        prediction = this.neuralNetwork.feedForward(data.inputs);
        $predictionContainer.find('td#target_' + joinedInputs).text(data.outputs[0]);
        _results.push($predictionContainer.find('td#prediction_' + joinedInputs).text(_.num.round(prediction, 2)));
      }
      return _results;
    };

    App.prototype.run = function() {
      var self;
      self = this;
      this.epoch = 0;
      return this.timer = setInterval(function() {
        var data, trainingItem, _i, _len, _ref;
        trainingItem = _.sample(self.trainingData);
        self.neuralNetwork.train(trainingItem.inputs, trainingItem.outputs);
        self.epoch++;
        if (self.epoch % 10 === 0) {
          self.render();
          console.log(self.epoch);
          self.updateSparkline();
          self.updatePredictionDisplay();
        }
        if (false) {
          _ref = self.trainingData;
          for (_i = 0, _len = _ref.length; _i < _len; _i++) {
            data = _ref[_i];
            console.log(data.inputs, 'Prediction: ' + self.neuralNetwork.feedForward(data.inputs));
          }
          return clearInterval(self.timer);
        }
      }, 0);
    };

    App.prototype.pause = function() {
      clearInterval(this.timer);
      return delete this.timer;
    };

    return App;

  })();

  $(function() {
    var app, canvasHeight, controlsHeight, trainingData;
    resizeCanvas();
    canvasHeight = $('canvas').height();
    controlsHeight = $('div#controls').height();
    $('div#controls').css('top', (canvasHeight - controlsHeight) / 2);
    trainingData = {
      and: [
        {
          inputs: [0, 0],
          outputs: [0]
        }, {
          inputs: [0, 1],
          outputs: [0]
        }, {
          inputs: [1, 0],
          outputs: [0]
        }, {
          inputs: [1, 1],
          outputs: [1]
        }
      ],
      or: [
        {
          inputs: [0, 0],
          outputs: [0]
        }, {
          inputs: [0, 1],
          outputs: [1]
        }, {
          inputs: [1, 0],
          outputs: [1]
        }, {
          inputs: [1, 1],
          outputs: [1]
        }
      ],
      nand: [
        {
          inputs: [0, 0],
          outputs: [1]
        }, {
          inputs: [0, 1],
          outputs: [1]
        }, {
          inputs: [1, 0],
          outputs: [1]
        }, {
          inputs: [1, 1],
          outputs: [0]
        }
      ],
      xor: [
        {
          inputs: [0, 0],
          outputs: [0]
        }, {
          inputs: [0, 1],
          outputs: [1]
        }, {
          inputs: [1, 0],
          outputs: [1]
        }, {
          inputs: [1, 1],
          outputs: [0]
        }
      ]
    };
    app = new App;
    app.render();
    app.targetSetName = 'xor';
    app.trainingData = trainingData.xor;
    app.updatePredictionDisplay();
    $('div#targets a').on('click', function(event) {
      var target;
      event.preventDefault();
      $('div#choose a').removeClass('target');
      $(this).addClass('target');
      target = $(this).attr('id');
      app.targetSetName = target;
      console.log('Setting target to ' + target);
      return app.trainingData = trainingData[target];
    });
    $('a#start').on('click', function(event) {
      event.preventDefault();
      if ($(this).text() === 'Start') {
        $(this).text('Pause');
        app.run();
        return $('div#summary').css('opacity', 1);
      } else {
        $(this).text('Start');
        return app.pause();
      }
    });
    return $(document).on('keydown', function(event) {
      if (KEY_SPACE === event.which) {
        $('a#start').click();
        return event.preventDefault();
      }
    });
  });

}).call(this);
