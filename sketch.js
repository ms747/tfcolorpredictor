let data;
let train_inputs;
let test_inputs;
let train_labels;
let test_labels;
let inputs = [];
let labels = [];
let model;
let r = 100;
let b = 100;
let g = 100;

function preload() {
	loadJSON("data.json", (dat) => {
		data = dat.colors
	})
}

function setup() {
	createCanvas(400, 400);
	const keys = Object.keys(data);
	for (key of keys) {
		inputs.push([data[key].r / 255, data[key].g / 255, data[key].b / 255])
		labels.push([(data[key].type === "light" ? 1 : 0)])
	}

	[train_inputs, test_inputs] = tf.tidy(() => {
		return tf.split(tf.tensor2d(inputs), [90, 23])
	});

	[train_labels, test_labels] = tf.tidy(() => {
		return tf.split(tf.tensor2d(labels), [90, 23])
	});

	/* Model */
	model = tf.sequential()
	model.add(tf.layers.dense({
		units: 64,
		inputShape: [3],
		activation: 'relu'
	}))
	model.add(tf.layers.dense({
		units: 64,
		inputShape: [64],
		activation: 'relu'
	}))
	model.add(tf.layers.dense({
		units: 1,
		inputShape: [64],
	}))

	model.compile({
		optimizer: tf.train.adam(0.05),
		loss: tf.losses.meanSquaredError
	})

	model.fit(train_inputs, train_labels, {
		epochs: 300,
		shuffle: true,
	}).then((hist) => {
		test_inputs.dispose()
		test_labels.dispose()
		train_inputs.dispose()
		train_labels.dispose()
		console.log("Loss after training", hist.history.loss[hist.history.loss.length - 1])
	})


}

function predColor(red, green, blue) {
	r = red
	g = green
	b = blue
	const input = tf.tensor2d([red / 255, green / 255, blue / 255], [1, 3])
	const p = model.predict(input)
	if (p.dataSync()[0] > 0.5) {
		p.dispose()
		input.dispose()
		return "light";
	} else {
		p.dispose()
		input.dispose()
		return "dark";
	}
}

function pred(id) {
	console.log(data[id].type)
	const input = tf.tensor2d([data[id].r / 255, data[id].g / 255, data[id].b / 255], [1, 3])
	const p = model.predict(input)
	if (p.dataSync()[0] > 0.5) {
		p.dispose()
		input.dispose()
		return "light";
	} else {
		p.dispose()
		input.dispose()
		return "dark";
	}
}

function draw() {
	background(r, g, b)
}