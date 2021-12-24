<?php
// var_dump($_POST);
// Fetching variables of the form which travels in URL
$neighborhood = $_POST['neighborhood'];
$filename = str_replace(" ","_",$neighborhood);
$filename = str_replace("-","_",$filename);
$filename = str_replace("+","_",$filename);
echo $filename.".html";
if($neighborhood !=''&& $neighborhood !='placeholder')
{
//  To redirect form on a particular page
header("Location:".$filename.".html");
}
else{
?><span><?php echo "Please select a neighborhood!";?></span> <?php
}
?>