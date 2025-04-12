// Add your CSS rules or selectors here
var css = ` 
.stream-selection .label {
    font-weight: bold;
    color: #ff4e33; /* Adjust the color as needed */
}

.stream-selection .question {
    display: flex;
    align-items: center; /* Aligns items vertically in the center */
    justify-content: flex-start; /* Aligns items horizontally to the start */
    margin-bottom: 2px; /* Adjust spacing as needed */
}

.stream-selection .question-label {
    font-weight: bold;
    color: #ff4e33; /* Adjust the color as needed */
    margin-right: 10px;
    width: 150px; /* Adjust as needed for the longest label */
}

.stream-selection .measure-container {
    display: flex;
    align-items: center; /* Aligns items vertically in the center */
    margin-bottom: 6px; /* Adjust spacing as needed */
}

.stream-selection .measure-label {
    font-weight: bold;
    color: #232323; /* Adjust the color as needed */
    margin-right: 10px; /* Space between label and control */
}

.stream-selection .spotfire-button {
    width: 80%; /* Percentage width for flexibility */
    padding-top: 3px;
    padding-bottom: 3px;
    font-size: 12px; /* Smaller font size */
    background: #50535b; /* Darker background color */
    font-family: "Century Gothic", sans-serif; /* Specific font family */
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.3s;
}

.stream-selection .spotfire-button:hover {
    background-color: #4199b8; /* Change color on hover */
    transform: scale(1.05);
}
`;

// Inject the CSS into an HTML tag which has the id StyleDiv
$("<style/>").text(css).appendTo($("#styleDiv"));
