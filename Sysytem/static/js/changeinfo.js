function changeInfo(_this) {
    // console.log(_this);//_this is the current node
    var parentNode = _this.parentNode.parentNode;//get the parentnode <div class="popContent">
    let $parentNode = $(parentNode);
    $.ajax({
        url: '/changeInfo',
        data: {
            //return the selected node's value
            asset_code: $parentNode.find('#asset_code').val(),
            name: $parentNode.find('#name').val(),
            num: $parentNode.find('#num').val(),
            lat: $parentNode.find('#lat').val(),
            lng: $parentNode.find('#lng').val(),
            storeadd:$parentNode.find('#storeadd').val(),
            description: $parentNode.find('#description').val()
        },
        dataType: 'JSON',
        type: 'GET',
        success: function () {
            console.log("success")
        }
    });
    history.go(0);// refresh the page
}
