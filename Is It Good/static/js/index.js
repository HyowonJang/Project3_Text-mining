$(document).ready(function(){
  var isLoading = false;

  var predicted_data = {};

  $('.prediction').click(function(){
    if(isLoading){
      alert("로딩중. 보통 1분 걸림.");
      return;
    }
    isLoading = true;
    // document가 준비되고 pred
    var link = $('.link').val();
    // "input row" div안의 input tag에 입력된 값을 link라는 변수에 받는다
    // console.log(link);
    var product = link.split('?')[0].split('/')[5];
    var url = "/predict?link=" + product;
    // console.log("url", url);
    // getJson이 frontend와 server을 연결
    // like라는 key에 product라는 value를 담아 만든 url을 서버로 보내서 결과를 받아온다
    $.getJSON(url, function(data){
      console.log(data);
      predicted_data = data;
      isLoading = false;
      $('.alert').empty();
        var item_name = "<p> 상품명 : " + data.item_name + "</p>";
        var review_total_cnt = "<p> 상품평 총 개수 : " + data.review_total_cnt + "</p>";
        var valid_review_cnt = "<p> 분석가능한 상품평 개수 : " + data.valid_review_cnt + "</p>";
        var total = item_name + review_total_cnt + valid_review_cnt;
        $('.alert').append(total);
        // append - list append X, html에 보여주는 내용 추가

      for(var i =0; i <data.neg_or_pos.length; i++){
        var tag ="<p>"+ data.neg_or_pos[i] + " : " + data.neg_or_pos_cnt[i] + "</p>";
        $('.alert').append(tag);
      }

      var chart_series = [];
      for(var i = 0; i < data.neg_or_pos.length; i++){
        chart_series.push({
          name: data.neg_or_pos[i],
          y: data.neg_or_pos_cnt[i]
        })
      }
      console.log(chart_series);

      draw_chart(chart_series);

      var neg_top_word = [];
      var columns = ["#", "negative words", "count", "button"]; //, "review"
      var html_str = "<table class='table'>";
      html_str += "<thead class='thead-dark'>";
      html_str += "<tr>";
      for(var i = 0; i < columns.length; i++){
        html_str += "<th scope='col'>" + columns[i] + "</th>"
      }
      // thead, tbody
      // th : table head
      // tr : table row
      // td : table data

      html_str += "</tr>"
      html_str += "</thead>"

      html_str += "<tbody>"

      var btn_front = "<button type=\"button\" class=\"btn btn-primary btn-to-show-list\" n-data='";
      var btn_back = "' data-toggle=\"modal\" data-target=\"#exampleModalLong\">상품평 보기</button>";

      for(var i = 0; i < data.neg_result.word.length; i++){
        html_str += "<tr class='table-data'>"
        html_str += "<th scope='row'>" + (i+1) + "</th>"
        html_str += "<td>" + data.neg_result.word[i] + "</td>"
        html_str += "<td>" + data.neg_result.count[i] + "</td>"
        html_str += "<td>" + btn_front + i + btn_back + "</td>"
        html_str += "</tr>"
      }
      html_str += "</tbody>"

      html_str += "</table>"

      $('#table_negative').html(html_str)

      var pos_top_word = [];
      var columns = ["#", "positive words", "count", "button"]; //, "review"
      var html_str = "<table class='table'>";
      html_str += "<thead class='thead-dark'>"
      html_str += "<tr>"
      for(var i = 0; i < columns.length; i++){
        html_str += "<th scope='col'>" + columns[i] + "</th>"
      }
      html_str += "</tr>"
      html_str += "</thead>"

      html_str += "<tbody>"


      for(var i = 0; i < data.pos_result.word.length; i++){
        html_str += "<tr>"
        html_str += "<th scope='row'>" + (i+1) + "</th>"
        html_str += "<td>" + data.pos_result.word[i] + "</td>"
        html_str += "<td>" + data.pos_result.count[i] + "</td>"
        html_str += "<td>" + btn_front + i + btn_back + "</td>"
        html_str += "</tr>"
      }
      html_str += "</tbody>"

      html_str += "</table>"

      $('#table_positive').html(html_str)

    });
    // getJSON 함수 종료


  });
  // click 함수 종료

  $('#table_negative').on('click', '.btn-to-show-list', function() {
    var index = $(this).attr('n-data');
    var reviews = predicted_data.neg_result.review[index];
    console.log(reviews);
    var str = "";
    for(var i = 0; i < reviews.length; i ++){
      str += "<p style='padding:5px; border:1px solid #333; border-radius: 4px;'>" + reviews[i] + "</p>";
    }

    $('.contents-body-neg').html(str);
  });

  $('#table_positive').on('click', '.btn-to-show-list', function() {
    var index = $(this).attr('n-data');
    var reviews = predicted_data.pos_result.review[index];
    console.log(reviews);
    var str = "";
    for(var i = 0; i < reviews.length; i ++){
      str += "<p style='padding:5px; border:1px solid #333; border-radius: 4px;'>" + reviews[i] + "</p>";
    }

    $('.contents-body-pos').html(str);
  });

});
// ready 함수 종료

function draw_chart(series){
// Build the chart
  Highcharts.chart('container', {
    chart: {
        plotBackgroundColor: null,
        plotBorderWidth: null,
        plotShadow: false,
        type: 'pie'
    },
    title: {
        text: '상품평 분석'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            cursor: 'pointer',
            dataLabels: {
                enabled: false
            },
            showInLegend: true
        }
    },
    series: [{
        name: 'Brands',
        colorByPoint: true,
        data: series
    }]
  });
}
