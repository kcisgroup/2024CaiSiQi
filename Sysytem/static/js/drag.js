jQuery(document).ready(function () {
    var src_posi_Y = 0, dest_posi_Y = 0, move_Y = 0, is_mouse_down = false, destHeight = 200;
    $("#expander").mousedown(function (e) {
        src_posi_Y = e.pageY;//鼠标指针的位置
        is_mouse_down = true;
    });
    $(document).bind("click mouseup", function (e) {
        if (is_mouse_down) {
            is_mouse_down = false;
        }
    }).mousemove(function (e) {
        dest_posi_Y = e.pageY;
        move_Y = src_posi_Y - dest_posi_Y;
        src_posi_Y = dest_posi_Y;
        destHeight = $("#topRows").height() - move_Y;
        if (is_mouse_down) {
            $("#topRows").css("height", destHeight > 100 ? destHeight : 100);
        }
    });
});

jQuery(document).ready(function () {
    var src_posi_Y = 0, dest_posi_Y = 0, move_Y = 0, is_mouse_down = false, destHeight = 200;
    $("#expander1").mousedown(function (e) {
        src_posi_Y = e.pageY;//鼠标指针的位置
        is_mouse_down = true;
    });
    $(document).bind("click mouseup", function (e) {
        if (is_mouse_down) {
            is_mouse_down = false;
        }
    }).mousemove(function (e) {
        dest_posi_Y = e.pageY;
        move_Y = src_posi_Y - dest_posi_Y;
        src_posi_Y = dest_posi_Y;
        destHeight = $("#topRows1").height() - move_Y;
        if (is_mouse_down) {
            $("#topRows1").css("height", destHeight > 100 ? destHeight : 100);
        }
    });
});