var outerWidth = 300;
var outerHeight = 250;
var circleRadius = 5;

var xColumn = "sepal_length";
var yColumn = "petal_length";

var svg = d3.select("body").append("svg")
.attr("width", outerWidth)
.attr("height", outerHeight);

var xScale = d3.scale.linear().range([0, outerWidth]);
var yScale = d3.scale.linear().range([outerHeight, 0]);

function render(data){
xScale.domain(d3.extent(data, function (d){ return d[xColumn]; }));
yScale.domain(d3.extent(data, function (d){ return d[yColumn]; }));
var circles = svg.selectAll("circle").data(data);
circles.enter().append("circle").attr("r", circleRadius);
circles
  .attr("cx", function (d){ return xScale(d[xColumn]); })
  .attr("cy", function (d){ return yScale(d[yColumn]); });
circles.exit().remove();
}

function type(d){
d.sepal_length = +d.sepal_length;
d.sepal_width  = +d.sepal_width;
d.petal_length = +d.petal_length;
d.petal_width  = +d.petal_width;
return d;
}
d3.csv("ML_Datasets/iris.csv", type, render);