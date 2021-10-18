/*!
* Start Bootstrap - Creative v7.0.4 (https://startbootstrap.com/theme/creative)
* Copyright 2013-2021 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-creative/blob/master/LICENSE)
*/
//
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    // Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            offset: 74,
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

    // Activate SimpleLightbox plugin for portfolio items
    new SimpleLightbox({
        elements: '#portfolio a.portfolio-box'
    });

});

function getDate(){
    var date = new Date()
    $("#times").text(date.toLocaleString())
}

$('document').ready(async () => {

    setInterval("getDate()",1000);

    const url = 'temp'
    let tmp = null
    checkInterval = setInterval(async () => {
        //console.log('fresh_env')
        await fetch(url).then(async response => {
            tmp = await response.text()
            let tmmp = tmp.replace('*', '°').split('\n')
            $('#temp').text(tmmp[0])
            $('#humi').text(tmmp[1])
        })
    }, 1500)

    $('input[value="History"]').click(() => {
        $('table[name="history"]').toggleClass('d-none')
        let checkInterval = null
        if(!$('table[name="history"]').hasClass('d-none')) {
            const url = 'show'
            let data = null
            checkInterval = setInterval(async () => {
                //console.log('fresh')
                await fetch(url).then(response => response.json())  
                .then(json => {
                    data = json
                })
                $('tbody').empty()
                for(let i = Object.keys(data).length-1; i >= Object.keys(data).length - 10; i--) {
                    let html = null
                    if(data[i].name === '')
                        html = `<tr>
                                    <td><span class="text-danger">Stranger</span></td>
                                    <td><span class="text-danger">${data[i].time}</span></td>
                                </tr>`
                    else
                        html = `<tr>
                                    <td>${data[i].name}</td>
                                    <td>${data[i].time}</td>
                                </tr>`
                    $('tbody').append(html)
                }
                if($('table[name="history"]').hasClass('d-none'))
                    clearInterval(checkInterval)
            }, 3000)
        }
    })

    $('input[value="Init"]').click(() => {
        $.ajax({
            type: 'POST',
            contentType: 'application/json',
            dataType: 'json',
            url: '/init/lamb/guess/31',
            success: function () {
                $('#exampleModal').modal('show')
            }
        });
    })

    $('input[value="Train"]').click(() => {
        $.ajax({
            type: 'POST',
            contentType: 'application/json',
            dataType: 'json',
            url: '/train',
            success: function () {
                $('#exampleModal').modal('show')
            }
        });
    })

    $('input[value="Environment"]').click(() => {
        $('table[name="env"]').toggleClass('d-none')
    })

    $('#modal_close').click(() => {
        $('#exampleModal').modal('hide')
    })
})

$(document).on({
    ajaxStart: function() {$("body").addClass("loading")},
    ajaxStop: function() {$("body").removeClass("loading")}    
});


