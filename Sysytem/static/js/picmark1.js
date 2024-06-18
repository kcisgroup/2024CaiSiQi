$(document).ready(function () {
  var url = '../static/js/markdata.json';
  $.getJSON(url, function (data) {
    // console.log(data['image_1']);
    // lat = data['image_1'][0].coords.lat;
    $('.pin').easypinShow({
      data: data,
      responsive: false,
      popover: {
        show: false,
        animate: true
      },
      each:function(index, data) {
        data.longitude = data.coords.long;
        data.latitude = data.coords.lat;

        // console.log("in",index)
        // console.log("da",data)
        return data;
      },
      error: function (e) {
        console.log(e);
      },
      success: function () {
        console.log('success');
      }
    });
  })
});