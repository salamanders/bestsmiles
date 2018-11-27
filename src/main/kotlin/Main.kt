import net.tzolov.cv.mtcnn.MtcnnService
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Point
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

fun main() {
    println(System.getProperty("java.version"))
    println(System.getProperty("os.name"))
    println(System.getProperty("os.version"))
    println(System.getProperty("os.arch"))

    val mtcnnService = MtcnnService()
    val inputImage = ImageIO.read(File("DSC02953_cropped.jpg"))
            .to3ByteBGR()

    println("Loaded image as BGR: ${inputImage.width}x${inputImage.height} ${inputImage.type}")
    val tileMap = inputImage.toOverlappingTiles()
    println("Working with ${tileMap.size} tiles")

    val faces = tileMap.map { (location, image) ->

        val faceAnnotations = mtcnnService.faceDetection(image)!!
        println("${location.x}x${location.y} = ${faceAnnotations.size}")
        location to faceAnnotations.filterNotNull()
    }

    println("Found all faces with overlap: Tiles: ${faces.size}, Faces: ${faces.sumBy { it.second.size }}")
    inputImage.createGraphics().let { g2d ->
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g2d.stroke = BasicStroke(3.0f)
        g2d.color = Color.YELLOW

        faces.forEach { (offset, fas) ->
            fas.forEach { fa ->
                g2d.drawRect(offset.x + fa.boundingBox.x, offset.y + fa.boundingBox.y, fa.boundingBox.w, fa.boundingBox.h)
            }
        }
        g2d.dispose()
    }
    ImageIO.write(inputImage, "png", File("allFaces.png"))
}


// Use MtcnnUtil in next version
fun BufferedImage.to3ByteBGR(): BufferedImage {
    if (type == BufferedImage.TYPE_3BYTE_BGR) {
        return this
    }
    val outputImage = BufferedImage(width, this.height, BufferedImage.TYPE_3BYTE_BGR)
    outputImage.graphics.drawImage(this, 0, 0, null)
    return outputImage
}

fun BufferedImage.toOverlappingTiles(res: Int = 400): Map<Point, BufferedImage> {
    return (0..(height - res) step res / 2).flatMap { y ->
        (0..(width - res) step res / 2).map { x ->
            Point(x, y)
        }
    }.map { point ->
        point to getSubimage(point.x, point.y, res, res)
    }.toMap()
}