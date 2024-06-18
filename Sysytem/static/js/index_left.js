var checkValues = [];

function clickCheckbox() {
    var checkDomArr = document.querySelectorAll('.multi-table tbody input[type=checkbox]:checked');
    checkValues = [];
    for (var i = 0, len = checkDomArr.length; i < len; i++) {
        checkValues.push(checkDomArr[i].value);
    }
    updateText();
    var allCheckDomArr = document.querySelectorAll('.multi-table tbody input[type=checkbox]');
    var allCheckbox = document.getElementById('js-all-checkbox');
    for (var i = 0, len = allCheckDomArr.length; i < len; i++) {
        if (!allCheckDomArr[i].checked) {
            if (allCheckbox.checked)
                allCheckbox.checked = false;
            break;
        } else if (i === len - 1) {
            document.getElementById('js-all-checkbox').checked = true;
            return;
        }
    }
}

function checkAll(current) {
    var allCheckDomArr = document.querySelectorAll('.multi-table tbody input[type=checkbox]');
    if (!current.checked) { // 点击的时候, 状态已经修改, 所以没选中的时候状态时true
        checkValues = [];
        for (var i = 0, len = allCheckDomArr.length; i < len; i++) {
            var checkStatus = allCheckDomArr[i].checked;
            if (checkStatus)
                allCheckDomArr[i].checked = false;
        }
    } else {
        checkValues = [];
        for (var i = 0, len = allCheckDomArr.length; i < len; i++) {
            var checkStatus = allCheckDomArr[i].checked;
            if (!checkStatus)
                allCheckDomArr[i].checked = true;
            checkValues.push(allCheckDomArr[i].value);
        }
    }
    updateText();
}

function updateText() {
    // document.getElementById('js-check-text').innerHTML = JSON.stringify(checkValues);
    // console.log(checkValues);
    $.ajax({
        url: '/mark',
        data: {
            values: JSON.stringify(checkValues),
        },
        dataType: 'JSON',
        type: 'GET',
        success: function (data) {
            console.log(data);
//            window.parent.frames["down"].location.reload();
            if (window.parent.frames && window.parent.frames["down"]) {
                window.parent.frames["down"].location.reload();
            }
            console.log(data.length)
                    var html = "";
                    for (let i = 0; i < data.length; i++) {
                        // console.log(data[i]);
                        html = html + "<tr><td>" + data[i]['资产编码'] + "</td><td>" +data[i]['设备名称'] +
                            "</td><td>" + data[i]['同类型总数量'] + "</td><td>" + data[i]["其他描述/取值范围"] +
                            "</td><td>" + data[i]['存放安装地点'] + "</td><td>" + data[i]['一级'] + "</td><td>"
                            + data[i]['二级'] + "</td></tr>"
                    }
                    // console.log(html)
                    document.getElementById("tbody1").innerHTML = html;

        }
    });
}

$(document).ready(function () {
    fetch('../static/js/choose_data.json')
        .then((res) => res.json())
        .then((data) => {
            // console.log(data[0]["一级"]);
            var html = "";
            for (let i = 0; i < data.length; i++) {
                html = html + "<tr><td><input type='checkbox' name='select' value='" + data[i]["资产编码"] + "' onclick='clickCheckbox()'></td>" +
                    "<td>" + data[i]["存放安装地点"] + "</td><td>" + data[i]["一级"] + "</td><td>" + data[i]["二级"] + "</td><td>" + data[i]["资产编码"] + "</td><td>" + data[i]["设备名称"] + "</td>" +
                    "<td>" + data[i]["同类型总数量"] + "</td><td>" + data[i]["其他描述/取值范围"] + "</td><td>" + data[i]["单位"] + "</td><td>" + data[i]["原值"] + "</td></tr>"
            }
            // console.log(html)
            document.getElementById("tbody").innerHTML = html;
        });

});