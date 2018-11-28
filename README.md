

Based on library and sample in https://github.com/tzolov/mtcnn-java/blob/master/src/test/java/net/tzolov/cv/mtcnn/sample/FaceDetectionSample1.java

400px squares = 
308 faces
3.7 minutes


200px squares =
314 faces
1.5 minutes


100px sq =
302 faces
0.7 minutes
 
    inputImage.createGraphics()!!.let { g2d ->
    g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
    g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g2d.stroke = BasicStroke(3.0f)
    g2d.color = Color.YELLOW
    
    faces.forEach {
        g2d.drawRect(it.boundingBox.x, it.boundingBox.y, it.boundingBox.w, it.boundingBox.h)
    }
    g2d.dispose()
    }